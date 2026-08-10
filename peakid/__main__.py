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
import sys

import requests

from src.dem import TileNotFoundError, elevation_m, locate_tile_path
from src.geo import azimuth_deg, destination_point_deg, elevation_deg, haversine_m
from src.horizon import Visibility, check_visibility, horizon_profile
from src.peaks import visible_peaks
from src.render import render_horizon_png

EXIT_DATA = 2      # falta un dato (tile, void en el observador)
EXIT_NET = 3       # Overpass no responde
EXIT_UNIMPL = 4    # funcionalidad reservada sin implementar (--photo)


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
    _add_observer_args(panorama)
    _add_peaks_args(panorama)
    panorama.add_argument("-o", "--out", default="panorama.png",
                          help="fichero PNG de salida (defecto: panorama.png)")
    panorama.add_argument("--photo", metavar="FICHERO",
                          help="[reservado, fase 2] alinear con una fotografía")
    panorama.set_defaults(handler=_cmd_panorama)

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


def _add_observer_args(p: argparse.ArgumentParser) -> None:
    # --lat/--lon con nombre, NUNCA posicionales: una longitud oeste
    # posicional ("-4.097") se parsearía como opción desconocida, y en España
    # ese es el caso normal, no un borde.
    p.add_argument("--lat", type=_lat, required=True, help="latitud del observador")
    p.add_argument("--lon", type=float, required=True, help="longitud del observador")
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


def _lat(text: str) -> float:
    value = float(text)
    if not -90.0 <= value <= 90.0:
        raise argparse.ArgumentTypeError(f"latitud fuera de rango: {value}")
    return value


def _cmd_panorama(args: argparse.Namespace) -> int:
    if args.photo:
        print("el alineamiento con fotografía no está implementado todavía "
              "(fase 2); --photo queda reservado", file=sys.stderr)
        return EXIT_UNIMPL
    eye_m, eye_desc = _eye_height(args.lat, args.lon, args.eye, args.dem_dir)
    sightings = _sightings(args, eye_m)
    profile = horizon_profile(args.lat, args.lon, eye_m, dem_dir=args.dem_dir)
    report = render_horizon_png(profile, args.out, peaks=sightings)

    print(f"observador ({args.lat:.4f}, {args.lon:.4f}), ojo a {eye_desc}")
    _print_sightings_summary(sightings)
    print(f"etiquetados {len(report.placed)}, "
          f"descartados por colisión {len(report.discarded)}")
    print(f"imagen: {args.out}")
    if args.csv:
        _write_csv(args.csv, sightings)
        print(f"csv: {args.csv}")
    return 0


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
                  f"{s.visibility.value:<8} {s.peak.name}")
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
        writer.writerow(["nombre", "h_m", "fuente_altitud", "lat", "lon",
                         "azimut_deg", "dist_km", "elev_deg", "estado",
                         "truncado_km"])
        for s in sightings:
            writer.writerow([
                s.peak.name, f"{s.h_m:.1f}", s.h_source,
                f"{s.lat_deg:.5f}", f"{s.lon_deg:.5f}",
                f"{s.azimuth_deg:.2f}", f"{s.distance_m / 1000:.2f}",
                f"{s.elevation_deg:.3f}", s.visibility.value,
                "" if s.truncated_at_m is None else f"{s.truncated_at_m / 1000:.1f}",
            ])


if __name__ == "__main__":
    sys.exit(main())
