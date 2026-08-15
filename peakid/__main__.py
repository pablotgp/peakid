"""python -m peakid — punto de entrada del pipeline completo.

Capa de presentación: parsea argumentos, llama a src/* y formatea. Cero
geometría aquí. Solo stdlib.

Convenios de salida: km y grados solo en texto para el usuario (CLAUDE.md);
los CSV llevan cabecera con unidades explícitas. Errores esperables salen
como mensaje claro por stderr con código de salida propio, nunca como
traceback; un traceback siempre significa bug del motor, no error de uso.
"""

import argparse
import csv
import math
import sys

import requests

from src.dem import TileNotFoundError, elevation_m, locate_tile_path
from src.geo import azimuth_deg, destination_point_deg, elevation_deg, haversine_m
from src.horizon import Visibility, check_visibility, horizon_profile
from src.peaks import visible_peaks
from src.render import render_horizon_png

EXIT_DATA = 2      # falta un dato (tile, void en el observador, foto)
EXIT_NET = 3       # Overpass no responde


class _CliError(Exception):
    def __init__(self, message: str, code: int = EXIT_DATA):
        super().__init__(message)
        self.code = code


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except _CliError as err:
        print(f"error: {err}", file=sys.stderr)
        return err.code
    except TileNotFoundError as err:
        print(f"error: {err}; descárgalo o ajusta --dem-dir", file=sys.stderr)
        return EXIT_DATA
    except requests.RequestException as err:
        print(f"error: Overpass no responde ({err}). Reintenta más tarde; "
              "las respuestas previas siguen cacheadas", file=sys.stderr)
        return EXIT_NET
    except KeyboardInterrupt:
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="peakid",
        description="Identificación geométrica de cimas: qué se ve y cómo se llama.")
    sub = parser.add_subparsers(dest="command", required=True)

    panorama = sub.add_parser(
        "panorama", help="barrido de 360° con picos etiquetados -> PNG")
    _add_observer_args(panorama, require_coords=False)  # --photo+GPS las suple
    _add_peaks_args(panorama)
    panorama.add_argument("-o", "--out", default="panorama.png",
                          help="fichero PNG de salida (defecto: panorama.png)")
    panorama.add_argument("--az-min", type=float, metavar="GRADOS",
                          help="recortar el dibujo desde este azimut "
                               "(con --az-max; puede cruzar el norte)")
    panorama.add_argument("--az-max", type=float, metavar="GRADOS",
                          help="recortar el dibujo hasta este azimut")
    panorama.add_argument("--photo", metavar="FICHERO",
                          help="abrir la herramienta de alineamiento manual "
                               "sobre esta fotografía (fase 2)")
    panorama.set_defaults(handler=_cmd_panorama)

    align = sub.add_parser(
        "align", help="alineamiento manual de una foto (fase 2)")
    _add_observer_args(align, require_coords=False)
    align.add_argument("--photo", metavar="FICHERO", required=True,
                       help="fotografía a alinear")
    align.add_argument("--search", action="store_true",
                       help="búsqueda automática previa por fuerza bruta; "
                            "abre la GUI con el mejor candidato precargado "
                            "para validarlo o corregirlo")
    align.add_argument("--az-hint", type=float, metavar="GRADOS",
                       help="azimut aproximado al que apuntaba la cámara; "
                            "centra ahí la búsqueda")
    align.add_argument("--az-margin", type=float, default=20.0,
                       metavar="GRADOS",
                       help="semirrango de azimut explorado (defecto: 20)")
    align.add_argument("--fov-hint", type=float, metavar="GRADOS",
                       help="campo de visión aproximado (útil con "
                            "teleobjetivo, donde el rango por defecto es "
                            "demasiado ancho para resolverlo)")
    align.add_argument("--fov-margin", type=float, default=10.0,
                       metavar="GRADOS",
                       help="semirrango de FOV explorado con --fov-hint "
                            "(defecto: 10)")
    align.add_argument("--radius-km", type=float, default=60.0,
                       help="radio de búsqueda de topónimos en km "
                            "(defecto: 60)")
    align.add_argument("--cache-dir", default="cache",
                       help="directorio de caché de Overpass")
    align.add_argument("--no-peaks", action="store_true",
                       help="no cargar topónimos (útil sin red y sin caché)")
    align.set_defaults(handler=_cmd_align_photo)

    peaks = sub.add_parser(
        "peaks", help="lista de picos visibles, sin imagen")
    _add_observer_args(peaks)
    _add_peaks_args(peaks)
    peaks.set_defaults(handler=_cmd_peaks)

    visible = sub.add_parser(
        "visible", help="¿se ve el objetivo desde el observador?")
    _add_observer_args(visible)
    visible.add_argument("--target-lat", type=_lat, required=True,
                         help="latitud del objetivo")
    visible.add_argument("--target-lon", type=float, required=True,
                         help="longitud del objetivo")
    visible.add_argument("--target-ele", type=float, metavar="M",
                         help="altitud del objetivo en m (defecto: DEM en el "
                              "punto; con un pico real, mejor su altitud oficial)")
    visible.set_defaults(handler=_cmd_visible)
    return parser


def _add_observer_args(p: argparse.ArgumentParser,
                       require_coords: bool = True) -> None:
    # --lat/--lon con nombre, NUNCA posicionales: una longitud oeste
    # posicional ("-4.097") se parsearía como opción desconocida, y en España
    # ese es el caso normal, no un borde.
    p.add_argument("--lat", type=_lat, required=require_coords,
                   help="latitud del observador")
    p.add_argument("--lon", type=float, required=require_coords,
                   help="longitud del observador")
    p.add_argument("--eye", type=float, metavar="M",
                   help="altitud del OJO en m sobre el nivel del mar "
                        "(defecto: cota DEM + 1.7)")
    p.add_argument("--dem-dir", default="data", help="directorio de tiles .hgt")


def _add_peaks_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--radius-km", type=float, default=100.0,
                   help="radio de búsqueda de picos en km (defecto: 100)")
    p.add_argument("--csv", metavar="FICHERO",
                   help="volcar la lista de picos a un CSV")
    p.add_argument("--cache-dir", default="cache",
                   help="directorio de caché de Overpass")


def _require_valid_coords(lat_deg, lon_deg, origen: str) -> None:
    """Barrera única: ninguna coordenada no finita o fuera de rango pasa de
    aquí, venga del CLI, del EXIF o de una sesión guardada.

    Sanear el parseo del EXIF no basta: la garantía tiene que estar donde el
    dato ENTRA al motor, porque un NaN que llegue por cualquier otra vía
    produciría resultados plausibles sin que nada falle."""
    for nombre, valor in (("latitud", lat_deg), ("longitud", lon_deg)):
        if valor is None or not math.isfinite(float(valor)):
            raise _CliError(f"{nombre} inválida ({valor!r}) procedente de "
                            f"{origen}: no es un número utilizable", 1)
    if not -90.0 <= float(lat_deg) <= 90.0:
        raise _CliError(f"latitud fuera de rango ({lat_deg}) desde {origen}", 1)
    if not -180.0 <= float(lon_deg) <= 180.0:
        raise _CliError(f"longitud fuera de rango ({lon_deg}) desde {origen}", 1)


def _lat(text: str) -> float:
    value = float(text)
    if not -90.0 <= value <= 90.0:
        raise argparse.ArgumentTypeError(f"latitud fuera de rango: {value}")
    return value


def _cmd_panorama(args: argparse.Namespace) -> int:
    if args.photo:
        return _cmd_align_photo(args)
    if args.lat is None or args.lon is None:
        raise _CliError("--lat y --lon son obligatorios "
                        "(salvo con --photo cuyo EXIF traiga GPS)", 1)
    if (args.az_min is None) != (args.az_max is None):
        raise _CliError("--az-min y --az-max se usan juntos o ninguno", 1)
    eye_m, eye_desc = _eye_height(args.lat, args.lon, args.eye, args.dem_dir)
    sightings = _sightings(args, eye_m)
    profile = horizon_profile(args.lat, args.lon, eye_m, dem_dir=args.dem_dir)
    report = render_horizon_png(profile, args.out, peaks=sightings,
                                az_min_deg=args.az_min,
                                az_max_deg=args.az_max)

    print(f"observador ({args.lat:.4f}, {args.lon:.4f}), ojo a {eye_desc}")
    _print_sightings_summary(sightings)
    if args.az_min is not None:
        arc = (args.az_max - args.az_min) % 360.0 or 360.0
        print(f"sector {args.az_min:.0f}°–{args.az_max:.0f}° "
              f"({arc:.0f}° de arco), {len(report.out_of_range)} picos fuera")
    print(f"etiquetados {len(report.placed)}, "
          f"descartados por colisión {len(report.discarded)}")
    print(f"imagen: {args.out}")
    if args.csv:
        _write_csv(args.csv, sightings)
        print(f"csv: {args.csv}")
    return 0


def _cmd_align_photo(args: argparse.Namespace) -> int:
    """panorama --photo: herramienta de alineamiento manual (fase 2)."""
    from pathlib import Path

    from src.align import (AlignmentParams, alignment_json_path,
                           load_session, seed_from_exif)

    if not Path(args.photo).exists():
        raise _CliError(f"no existe la foto: {args.photo}")
    session = load_session(args.photo)
    try:
        exif = seed_from_exif(args.photo)
    except Exception:
        exif = {}  # EXIF corrupto o ausente: semillas por defecto

    # observador: CLI > sesión anterior > GPS del EXIF
    if args.lat is not None and args.lon is not None:
        lat_deg, lon_deg, obs_source = args.lat, args.lon, "cli"
    elif session is not None:
        observer = session["observer"]
        lat_deg, lon_deg = observer["lat_deg"], observer["lon_deg"]
        obs_source = observer.get("source", "cli")
        print(f"observador de la sesión anterior: ({lat_deg:.5f}, {lon_deg:.5f})")
    elif "lat_deg" in exif and "lon_deg" in exif:
        lat_deg, lon_deg, obs_source = exif["lat_deg"], exif["lon_deg"], "exif-gps"
        print(f"observador desde EXIF GPS: ({lat_deg:.5f}, {lon_deg:.5f})")
    elif exif.get("gps_invalid"):
        raise _CliError(
            "la foto trae un bloque GPS vacío o corrupto (la cámara no llegó "
            "a fijar posición); pasa --lat y --lon", 1)
    else:
        raise _CliError("la foto no trae GPS en el EXIF; pasa --lat y --lon", 1)
    _require_valid_coords(lat_deg, lon_deg, obs_source)

    if args.eye is None and session is not None:
        eye_m = session["observer"]["eye_m"]
        eye_desc = f"{eye_m:.1f} m (sesión anterior)"
    else:
        eye_m, eye_desc = _eye_height(lat_deg, lon_deg, args.eye, args.dem_dir)
    print(f"ojo a {eye_desc}; calculando el barrido de horizonte…")
    profile = horizon_profile(lat_deg, lon_deg, eye_m, dem_dir=args.dem_dir)

    seeds = {"azimuth_deg": 0.0, "hfov_deg": 65.0,
             "pitch_deg": 0.0, "roll_deg": 0.0}
    sources = {name: "default" for name in seeds}
    for name in ("azimuth_deg", "hfov_deg"):
        if name in exif:
            seeds[name] = exif[name]
            sources[name] = "exif"

    if session is not None:
        # reanudar desde el ajuste guardado, no desde las semillas: perder
        # el trabajo al reabrir es inaceptable con decenas de fotos
        initial = AlignmentParams(**session["alignment"])
        print(f"sesión anterior cargada: {alignment_json_path(args.photo)}")
        print("  " + "  ".join(f"{n}={v:.2f}"
                               for n, v in session["alignment"].items()))
    else:
        initial = AlignmentParams(**seeds)
        print("semillas: " + "  ".join(
            f"{n}={seeds[n]:.1f} ({sources[n]})" for n in seeds))

    if getattr(args, "search", False):
        # asistente de fuerza bruta: precarga la GUI cerca, el usuario juzga
        az_source = ("sesión" if session is not None
                     else sources.get("azimuth_deg", "default"))
        found = _search_alignment(args, args.photo, profile, initial,
                                  az_source)
        if found is not None:
            initial = found

    sightings = _peaks_for_align(args, lat_deg, lon_deg, eye_m)

    from src.align.gui import run_align  # import tardío: tkinter solo aquí
    saved = run_align(args.photo, profile, lat_deg, lon_deg, eye_m,
                      initial, sources, obs_source, resume=session,
                      peaks=sightings)
    if saved is None:
        print("cerrado sin guardar")
    else:
        print(f"alineamiento guardado: {saved}")
    return 0


def _peaks_for_align(args, lat_deg, lon_deg, eye_m):
    """Topónimos para superponer en la GUI.

    Es la validación que de verdad zanja un alineamiento: ver qué nombre y
    qué altitud caen sobre cada bulto de la foto. Un fallo al traerlos NO
    debe impedir alinear — la herramienta sigue siendo usable sin nombres —,
    así que la red se trata como opcional y se avisa.
    """
    if getattr(args, "no_peaks", False):
        return []
    try:
        radius_m = getattr(args, "radius_km", 60.0) * 1000.0
        return visible_peaks(lat_deg, lon_deg, eye_m, radius_m=radius_m,
                             max_distance_m=radius_m,
                             dem_dir=args.dem_dir,
                             cache_dir=getattr(args, "cache_dir", "cache"))
    except (requests.RequestException, OSError) as err:
        print(f"aviso: no se pudieron cargar los topónimos ({err}); "
              "la ventana se abre sin etiquetas", file=sys.stderr)
        return []


def _search_alignment(args, photo_path, profile, initial, az_source="default"):
    """Búsqueda automática (heurística asistente, NO la fase 2): detecta la
    cresta con un filtro clásico y busca por fuerza bruta los parámetros que
    la encajan con el perfil. El resultado se valida SIEMPRE a mano."""
    from src.align import load_oriented_photo
    from src.align.search import detect_photo_skyline, search_alignment

    img, photo_np = load_oriented_photo(photo_path)
    cols, rows, valid = detect_photo_skyline(photo_np)
    if valid.sum() < 50:
        print("búsqueda: cresta no detectable (calima, contraluz o primer "
              "plano) — se abre con los valores iniciales")
        return None

    # SIN pista de azimut se barren los 360°. Centrar en la semilla por
    # defecto (0°) y explorar ±20° es peor que inútil: ese 0 no significa
    # nada y el verdadero casi nunca está ahí, así que la búsqueda devolvía
    # con aplomo un sector arbitrario (pasó con Sierra Nevada y con Nerja).
    center_az = args.az_hint
    if center_az is None and az_source in ("exif", "sesión"):
        center_az = initial.azimuth_deg
    print(f"búsqueda: cresta detectada en {int(valid.sum())} columnas "
          f"({valid.mean():.0%})"
          + ("…" if center_az is not None
             else "; sin pista de azimut -> barrido completo de 360°…"))
    result = search_alignment(
        profile.azimuths_deg, profile.elevations_deg,
        cols[valid], rows[valid], img.width, img.height,
        center_az_deg=center_az, az_margin_deg=args.az_margin,
        fov_hint_deg=args.fov_hint, fov_margin_deg=args.fov_margin)
    # el espacio explorado SIEMPRE se imprime: un fallo por "nunca miré ahí"
    # debe ser visible sin tener que adivinarlo
    print(f"  espacio explorado: azimut {result.az_range_deg[0]:.1f}–"
          f"{result.az_range_deg[1]:.1f}°, "
          f"FOV {result.fov_range_deg[0]:.1f}–{result.fov_range_deg[1]:.1f}°")
    if not result.candidates:
        print("búsqueda: ningún candidato encaja — se abre con los valores "
              "iniciales")
        return None

    for i, cand in enumerate(result.candidates, 1):
        p = cand.params
        print(f"  {i}. az={p.azimuth_deg:6.2f}  fov={p.hfov_deg:5.2f}  "
              f"pitch={p.pitch_deg:+5.2f}  roll={p.roll_deg:+5.2f}   "
              f"error {cand.error_px:5.1f} px ({cand.error_deg:.2f}°)  "
              f"cobertura {cand.coverage:.0%}"
              + (f"  ⚠ SATURADO ({cand.saturated_fraction:.0%} de columnas al "
                 "tope: el error no mide, solo dice 'muy mal')"
                 if cand.saturated else ""))
    print("  (orden por error en píxeles; en grados un FOV estrecho parece "
          "mejor de lo que es)")

    if result.ambiguous and result.alternative is not None:
        alt = result.alternative.params
        print("  AVISO: resultado AMBIGUO, no lo tomes como solución. Hay "
              f"otro óptimo a {abs((alt.azimuth_deg - result.candidates[0].params.azimuth_deg + 180) % 360 - 180):.0f}° "
              f"de distancia (az={alt.azimuth_deg:.1f}, fov={alt.hfov_deg:.1f}) "
              f"solo un {result.ambiguity_margin:.0%} peor.")
        print("         Con campo estrecho la firma del horizonte es pobre y "
              "hay muchos máximos casi iguales; acota con --az-hint.")
    if result.fov_at_edge:
        print("  AVISO: el FOV óptimo se apoya en el borde del rango "
              "explorado; el verdadero puede estar FUERA. Usa --fov-hint.")
    if result.az_at_edge:
        print("  AVISO: el azimut óptimo se apoya en el borde del rango "
              "explorado; el verdadero puede estar FUERA. Usa --az-hint "
              "o amplía --az-margin.")
    if result.reliable:
        print("  sin reservas: el óptimo es claro y no toca ningún borde.")
    return result.candidates[0].params


def _cmd_peaks(args: argparse.Namespace) -> int:
    eye_m, eye_desc = _eye_height(args.lat, args.lon, args.eye, args.dem_dir)
    sightings = _sightings(args, eye_m)
    print(f"observador ({args.lat:.4f}, {args.lon:.4f}), ojo a {eye_desc}")
    _print_sightings_summary(sightings)
    if args.csv:
        _write_csv(args.csv, sightings)
        print(f"csv: {args.csv}")
    else:
        print(f"{'azimut':>7} {'dist':>8} {'elev':>7} {'altitud':>8}  "
              f"{'estado':<8} nombre")
        for s in sightings:
            print(f"{s.azimuth_deg:6.1f}° {s.distance_m / 1000:6.1f} km "
                  f"{s.elevation_deg:6.2f}° {s.h_m:6.0f} m  "
                  f"{s.visibility.value:<8} {s.peak.label}")
    return 0


def _cmd_visible(args: argparse.Namespace) -> int:
    eye_m, eye_desc = _eye_height(args.lat, args.lon, args.eye, args.dem_dir)
    if args.target_ele is not None:
        target_ele_m, target_desc = args.target_ele, "indicada con --target-ele"
    else:
        ground = elevation_m(args.target_lat, args.target_lon, args.dem_dir)
        if ground is None:
            raise _CliError("el DEM no tiene dato (void) en el objetivo "
                            f"({args.target_lat:.4f}, {args.target_lon:.4f}); "
                            "indica su altitud con --target-ele")
        target_ele_m, target_desc = ground, "del DEM"

    d_m = haversine_m(args.lat, args.lon, args.target_lat, args.target_lon)
    bearing_deg = azimuth_deg(args.lat, args.lon, args.target_lat, args.target_lon)
    target_elev_deg = elevation_deg(d_m, eye_m, target_ele_m)
    result = check_visibility(args.lat, args.lon, eye_m, args.target_lat,
                              args.target_lon, target_ele_m, dem_dir=args.dem_dir)

    print(result.status.value.upper())
    print(f"observador ({args.lat:.4f}, {args.lon:.4f}), ojo a {eye_desc}")
    print(f"objetivo   ({args.target_lat:.4f}, {args.target_lon:.4f}), "
          f"altitud {target_ele_m:.0f} m ({target_desc})")
    print(f"azimut {bearing_deg:.1f}°   distancia {d_m / 1000:.1f} km   "
          f"elevación {target_elev_deg:.2f}°")

    if result.status is Visibility.BLOCKED:
        bl_lat, bl_lon = destination_point_deg(args.lat, args.lon, bearing_deg,
                                               result.blocked_at_m)
        terrain_m = elevation_m(bl_lat, bl_lon, args.dem_dir)
        blocker_deg = elevation_deg(result.blocked_at_m, eye_m, terrain_m)
        print(f"obstáculo a {result.blocked_at_m / 1000:.1f} km "
              f"({bl_lat:.4f}, {bl_lon:.4f}): terreno a {terrain_m:.0f} m que "
              f"sube a {blocker_deg:.2f}°, por encima del objetivo "
              f"({target_elev_deg:.2f}°)")
    elif result.status is Visibility.UNKNOWN:
        cut_lat, cut_lon = destination_point_deg(args.lat, args.lon, bearing_deg,
                                                 result.truncated_at_m)
        tile = locate_tile_path(cut_lat, cut_lon)
        print(f"rayo truncado a {result.truncated_at_m / 1000:.1f} km: "
              f"falta {tile} en '{args.dem_dir}'")
    return 0


def _eye_height(lat_deg: float, lon_deg: float, eye_arg: float | None,
                dem_dir: str) -> tuple[float, str]:
    if eye_arg is not None:
        return eye_arg, f"{eye_arg:.1f} m (indicado con --eye)"
    ground_m = elevation_m(lat_deg, lon_deg, dem_dir)  # TileNotFoundError sube
    if ground_m is None:
        raise _CliError("el DEM no tiene dato (void) en el observador "
                        f"({lat_deg:.4f}, {lon_deg:.4f}); indica la altura "
                        "del ojo con --eye")
    return ground_m + 1.7, f"{ground_m + 1.7:.1f} m (cota DEM {ground_m:.1f} + 1.7)"


def _sightings(args: argparse.Namespace, eye_m: float):
    return visible_peaks(args.lat, args.lon, eye_m,
                         radius_m=args.radius_km * 1000.0,
                         dem_dir=args.dem_dir, cache_dir=args.cache_dir)


def _print_sightings_summary(sightings) -> None:
    unknown = sum(1 for s in sightings if s.visibility is Visibility.UNKNOWN)
    print(f"{len(sightings)} picos: {len(sightings) - unknown} visibles, "
          f"{unknown} sin confirmar (unknown)")


def _write_csv(path: str, sightings) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # nombre y alternativo en COLUMNAS SEPARADAS: en el CSV interesa
        # poder filtrar por cualquiera de los dos, no una cadena compuesta
        writer.writerow(["nombre", "alt_nombre", "h_m", "fuente_altitud",
                         "lat", "lon", "azimut_deg", "dist_km", "elev_deg",
                         "estado", "truncado_km"])
        for s in sightings:
            writer.writerow([
                s.peak.name, s.peak.alt_name or "", f"{s.h_m:.1f}", s.h_source,
                f"{s.lat_deg:.5f}", f"{s.lon_deg:.5f}",
                f"{s.azimuth_deg:.2f}", f"{s.distance_m / 1000:.2f}",
                f"{s.elevation_deg:.3f}", s.visibility.value,
                "" if s.truncated_at_m is None else f"{s.truncated_at_m / 1000:.1f}",
            ])


if __name__ == "__main__":
    sys.exit(main())
