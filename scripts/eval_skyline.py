"""Compara detectores de cresta contra los alineamientos manuales guardados.

No es un test: depende de fotos que no están versionadas. Se ejecuta a mano:

    python scripts/eval_skyline.py

Informa POR FOTO, sin promediar: con 5-6 casos una media escondería que una
empeora mientras otra mejora. El criterio de éxito es que ninguna empeore.

Lo que descalifica una referencia NO es haber usado la búsqueda automática,
sino haberla ACEPTADO SIN REVISAR. Verificar contra topónimos es información
independiente del detector, así que una referencia revisada así sirve como
verdad aunque el punto de partida saliera del buscador.

Categorías:

1. REVISADAS: alineadas a mano, o con ayuda automática y revisión humana
   posterior (`manual_review`). DECIDEN el veredicto.
2. ACEPTADAS SIN REVISAR: salida del buscador tal cual. Ahí el detector se
   mide contra su propia respuesta. Informativas, nunca dirimentes.
3. DUDOSAS (`low_confidence`): el propio alineamiento es de fiabilidad dudosa.
4. SIN PROCEDENCIA REGISTRADA: JSON anteriores al campo. No deciden, pero
   tampoco se descartan: hay que anotarles la revisión a mano.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.align import (AlignmentParams, infer_provenance,  # noqa: E402
                       load_oriented_photo, project_profile,
                       projected_y_per_column)
from src.align.search import search_alignment  # noqa: E402
from src.align.skyline import build_detector  # noqa: E402
from src.horizon import horizon_profile  # noqa: E402

DATASET = Path("Dataset")
# `dp` a secas sobre evidencia de color NO está: medido, empeora las cinco
# fotos de referencia (ver la cabecera de skyline.py).
DETECTORES = {nombre: build_detector(nombre)
              for nombre in ("heuristica", "modelo", "modelo+dp")}
# convenio de respaldo para JSON escritos antes de existir la casilla
MARCAS_DUDA = ("baja confianza", "dudos", "low_confidence")


def es_dudosa(alineamiento: dict) -> bool:
    if alineamiento.get("low_confidence"):
        return True
    notas = (alineamiento.get("notes") or "").lower()
    return any(marca in notas for marca in MARCAS_DUDA)


def evalua(photo_path: Path, alineamiento: dict) -> dict | None:
    manual = AlignmentParams(**alineamiento["alignment"])
    obs = alineamiento["observer"]
    try:
        img, photo = load_oriented_photo(str(photo_path))
    except OSError as err:
        print(f"  no se pudo abrir {photo_path.name}: {err}")
        return None
    width, height = img.size
    profile = horizon_profile(obs["lat_deg"], obs["lon_deg"], obs["eye_m"])

    # línea proyectada con los parámetros MANUALES: es la referencia contra
    # la que se mide el detector, aislado de cómo se comporte la búsqueda
    x_px, y_px, usable = project_profile(
        profile.azimuths_deg, profile.elevations_deg, manual, width, height)

    focal = (width / 2.0) / math.tan(math.radians(manual.hfov_deg) / 2.0)
    filas = {}
    for nombre, detector in DETECTORES.items():
        cols, rows, valid = detector(photo)
        if valid.sum() < 20:
            filas[nombre] = {"cobertura": float(valid.mean()), "sin_datos": True}
            continue
        c, r = cols[valid], rows[valid]
        y_line = projected_y_per_column(x_px, y_px, usable, c)
        mask = ~np.isnan(y_line)
        residuo = np.abs(r[mask] - y_line[mask])
        mediana = float(np.median(residuo)) if residuo.size else float("nan")

        resultado = search_alignment(
            profile.azimuths_deg, profile.elevations_deg, c, r, width, height,
            center_az_deg=manual.azimuth_deg, az_margin_deg=20.0)
        if resultado.candidates:
            az = resultado.candidates[0].params.azimuth_deg
            error_az = abs((az - manual.azimuth_deg + 180.0) % 360.0 - 180.0)
        else:
            az, error_az = float("nan"), float("nan")

        filas[nombre] = {
            "cobertura": float(valid.mean()),
            "residuo_px": mediana,
            "residuo_deg": math.degrees(math.atan(mediana / focal))
                           if math.isfinite(mediana) else float("nan"),
            "atipicos": float(np.mean(residuo > 3.0 * mediana))
                        if mediana > 0 else float("nan"),
            "az": az,
            "error_az": error_az,
            "dispersion_px": float(r.std()),
        }
    return filas


def main() -> int:
    alineamientos = sorted(DATASET.glob("*.align.json"))
    if not alineamientos:
        print(f"no hay alineamientos en {DATASET}/")
        return 1

    revisadas, sin_revisar, sin_registro, dudosas = [], [], [], []
    for ruta in alineamientos:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        foto = ruta.parent / datos["photo"]
        if not foto.exists():
            foto = Path(datos["photo"])          # junto al repo
        if not foto.exists():
            print(f"falta la foto de {ruta.name}: {datos['photo']}")
            continue
        filas = evalua(foto, datos)
        if filas is None:
            continue
        origen = infer_provenance(datos)
        etiqueta = datos["photo"]
        if origen.get("unknown"):
            etiqueta = "?" + etiqueta          # procedencia no registrada
        if es_dudosa(datos):
            dudosas.append((etiqueta, filas, origen))
        elif origen.get("unknown"):
            sin_registro.append((etiqueta, filas, origen))
        elif origen.get("reviewed"):
            revisadas.append((etiqueta, filas, origen))
        else:
            sin_revisar.append((etiqueta, filas, origen))

    def tabla(entradas, titulo):
        if not entradas:
            return
        print(f"\n{titulo}")
        print("  %-34s %-11s %7s %8s %7s %8s %7s" % (
            "foto", "detector", "cobert", "resid_px", "atip", "err_az", "desv"))
        for foto, filas, origen in entradas:
            for nombre, m in filas.items():
                if m.get("sin_datos"):
                    print("  %-34s %-11s %6.0f%%  sin columnas suficientes" % (
                        foto[:34], nombre, 100 * m["cobertura"]))
                    continue
                print("  %-34s %-11s %6.0f%% %8.1f %6.0f%% %7.2f° %7.0f" % (
                    foto[:34], nombre, 100 * m["cobertura"], m["residuo_px"],
                    100 * m["atipicos"], m["error_az"], m["dispersion_px"]))
            if origen.get("unknown"):
                print("       ↳ sin campo provenance; indicio: %s"
                      % origen.get("hint", "ninguno"))
                continue
            if origen.get("manual"):
                print("       ↳ manual: ninguna ayuda automática intervino")
                continue
            usado = origen.get("detector") or "detector no registrado"
            if origen.get("manual_review"):
                delta = origen.get("review_delta") or {}
                cuanto = (", ".join(f"{n} {v:+g}" for n, v in delta.items())
                          if delta else "sin delta numérico registrado")
                print("       ↳ búsqueda con %s, %s a mano (%s)"
                      % (usado,
                         (origen.get("review_kind") or "revisada").upper(),
                         cuanto))
                if origen.get("review_note"):
                    print("         %s" % origen["review_note"])
            else:
                print("       ↳ salida de %s ACEPTADA SIN REVISAR" % usado)

    tabla(revisadas, "REFERENCIAS REVISADAS (deciden el veredicto)")
    tabla(sin_revisar,
          "ACEPTADAS SIN REVISAR (el detector se mide contra su propia "
          "salida; informativas)")
    tabla(sin_registro,
          "SIN PROCEDENCIA REGISTRADA (anteriores al campo; informativas)")
    tabla(dudosas, "REFERENCIAS DUDOSAS (low_confidence, informativas)")

    fiables = revisadas
    if not fiables:
        print("\nNO HAY REFERENCIAS REVISADAS: no se emite veredicto.")
        print("  Revisar es mover parámetros después de la búsqueda, o "
              "verificar los topónimos y anotarlo en `provenance`.")
        print("  Para una referencia sin ayuda automática ninguna:")
        print("      python -m peakid align --photo FOTO --lat LAT --lon LON "
              "--no-auto-pitch-roll      (y sin --search)")
        return 0

    # SESGO DE ORIGEN: si el punto de partida de TODAS las referencias que
    # deciden lo produjo un mismo detector, el residuo en píxeles le favorece
    # por construcción por muy revisadas que estén — sobre todo donde la
    # revisión no corrigió ningún valor. Decirlo aquí, junto al veredicto, y
    # no solo en la documentación: es donde se va a leer.
    origenes = {o.get("detector") for _f, _m, o in fiables if o.get("search_used")}
    if len(origenes) == 1 and len(fiables) == len(
            [1 for _f, _m, o in fiables if o.get("search_used")]):
        unico = origenes.pop()
        # "verificacion" DECLARADO, no deducido de que review_delta esté
        # vacío: en 141720 la revisión corrigió (cambió de hipótesis entre
        # candidatos) y aun así no dejó delta numérico.
        sin_corregir = [f for f, _m, o in fiables
                        if o.get("review_kind") == "verificacion"]
        print(f"\nAVISO DE SESGO: las {len(fiables)} referencias que deciden "
              f"tienen su ORIGEN en '{unico}'.")
        print("  La revisión humana las valida como verdad, pero el residuo "
              "en PÍXELES sigue favoreciendo a ese detector por")
        print("  construcción: una diferencia del orden de 1 px en contra de "
              "otro NO es evidencia de que sea peor.")
        if sin_corregir:
            print("  Especialmente en las que se verificaron sin corregir "
                  "ningún valor (los números son los que propuso el buscador):")
            print("    " + ", ".join(f[:40] for f in sin_corregir))
        print("  Comparables hoy: err_az, cobertura y dispersión. El residuo "
              "lo será cuando haya referencias nacidas sin búsqueda.")

    for candidato in ("modelo", "modelo+dp"):
        print(f"\nveredicto {candidato} vs heuristica "
              "(solo referencias revisadas):")
        peor = None
        comparadas = 0
        for foto, filas, _origen in fiables:
            h, d = filas.get("heuristica", {}), filas.get(candidato, {})
            if not h or not d or h.get("sin_datos") or d.get("sin_datos"):
                continue
            comparadas += 1
            delta_res = d["residuo_px"] - h["residuo_px"]
            delta_az = d["error_az"] - h["error_az"]
            print("  %-34s residuo %+7.1f px   azimut %+6.2f°   %s" % (
                foto[:34], delta_res, delta_az,
                "mejora" if delta_res < 0 else "EMPEORA"))
            if delta_res > 0 and (peor is None or delta_res > peor[1]):
                peor = (foto, delta_res)
        if not comparadas:
            print("  (sin comparaciones: detector no disponible)")
        else:
            print("  -> " + ("ninguna empeora" if peor is None
                             else f"EMPEORA {peor[0]} en {peor[1]:.1f} px"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
