"""Tests dorados — contrato del proyecto.

NO modificar los valores esperados sin aprobación explícita del usuario
en el chat. Si un test falla, el bug está en el código (ver CLAUDE.md).
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.dem import (
    VOID,
    TileNotFoundError,
    elevation_m,
    elevations_m,
    load_tile,
    locate_tile_path,
)
from src.geo import (
    azimuth_deg,
    curvature_drop_m,
    destination_point_deg,
    destination_points_deg,
    elevation_deg,
    elevations_deg,
    haversine_m,
)
from src.horizon import Visibility, check_visibility, horizon_profile
from src.peaks import (
    Peak,
    _parse_ele,
    _ray_visibility,
    evaluate_peaks,
    fetch_peaks,
    relocate_peak,
)

# (lat_deg, lon_deg, altitud_m)
SOL = (40.4168, -3.7038, 650.0)
PENALARA = (40.8508, -3.9578, 2428.0)
BOLA_DEL_MUNDO = (40.7906, -3.9553, 2265.0)

DEM_DIR = "data"
PENALARA_TILE = Path(DEM_DIR) / "N40W004.hgt"  # ausente: proyecto movido a Málaga

# Mar abierto frente a Torre del Mar. Comprobado: las 2706 muestras del rayo
# al sur hasta el límite de cobertura son todas 0.0 (mar puro).
MAR = (36.730, -4.097)
MAROMA = (36.90194, -4.04444)  # altitud oficial 2069 m (Wikipedia)

PEAKFINDER_CSV = Path("tests/data/peakfinder_referencia.csv")
# El CSV no trae el punto de observación. Derivado de él: retrocediendo desde
# cada cima por su azimut inverso y su distancia, las 72 filas convergen a
# (36.7446, -4.0902) +-0.0003°; refinando para minimizar el peor residuo de
# azimut sale este punto, con error máximo 0.085° y RMS 0.040° sobre las 72.
# Que las coordenadas salgan redondas confirma que es el punto real y no un
# artefacto del ajuste. Dos parámetros libres contra 72 observaciones con
# residuos de 0.04° no pueden absorber un error sistemático (signo invertido,
# lat/lon cambiados), así que el ajuste no vuelve circular al test.
PEAKFINDER_OBS = (36.745, -4.090)


def test_curvatura():
    assert curvature_drop_m(10_000.0) == pytest.approx(6.8, rel=0.02)
    assert curvature_drop_m(30_000.0) == pytest.approx(61.5, rel=0.02)
    assert curvature_drop_m(60_000.0) == pytest.approx(245.9, rel=0.02)


def test_sol_penalara_distancia():
    d_m = haversine_m(SOL[0], SOL[1], PENALARA[0], PENALARA[1])
    assert d_m == pytest.approx(52_900.0, abs=1_000.0)


def test_sol_penalara_azimut():
    az = azimuth_deg(SOL[0], SOL[1], PENALARA[0], PENALARA[1])
    assert az == pytest.approx(336.0, abs=1.0)


def test_sol_penalara_elevacion():
    d_m = haversine_m(SOL[0], SOL[1], PENALARA[0], PENALARA[1])
    assert elevation_deg(d_m, SOL[2], PENALARA[2]) == pytest.approx(1.72, abs=0.05)


def test_sol_penalara_elevacion_sin_curvatura_control():
    # Control: si elevation_deg diera ~1.93, faltaría la corrección de
    # curvatura. El valor sin corregir se calcula inline a propósito:
    # no existe (ni debe existir) función pública sin curvatura.
    d_m = haversine_m(SOL[0], SOL[1], PENALARA[0], PENALARA[1])
    sin_curvatura_deg = math.degrees(math.atan2(PENALARA[2] - SOL[2], d_m))
    assert sin_curvatura_deg == pytest.approx(1.93, rel=0.02)
    assert elevation_deg(d_m, SOL[2], PENALARA[2]) < sin_curvatura_deg


def test_simetria_azimut():
    az_ab = azimuth_deg(SOL[0], SOL[1], PENALARA[0], PENALARA[1])
    az_ba = azimuth_deg(PENALARA[0], PENALARA[1], SOL[0], SOL[1])
    assert abs(((az_ab - az_ba) % 360.0) - 180.0) < 0.5


def test_sentido():
    # Norte: exacto en cualquier latitud (mismo meridiano).
    assert azimuth_deg(40.0, -3.0, 41.0, -3.0) == pytest.approx(0.0, abs=1e-9)
    # Este/oeste: solo exacto en el ecuador (los meridianos convergen).
    assert azimuth_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0, abs=1e-9)
    assert azimuth_deg(0.0, 0.0, 0.0, -1.0) == pytest.approx(270.0, abs=1e-9)


# --- src/dem/ -----------------------------------------------------------


def test_locate_tile_path():
    # floor, no int(): int(-4.10) truncaría a -4 -> "W004", que es el tile
    # vecino equivocado. floor(-4.10) = -5 -> "W005", la esquina SO real.
    assert locate_tile_path(36.74, -4.10) == "N36W005.hgt"
    assert locate_tile_path(40.85, -3.96) == "N40W004.hgt"


def test_load_tile_srtm1():
    tile = load_tile(str(Path(DEM_DIR) / "N38W004.hgt"))
    assert tile.n == 3600
    assert tile.array.shape == (3601, 3601)
    assert (tile.lat_sw, tile.lon_sw) == (38, -4)


def test_tile_ausente():
    # (0, -30): medio del Atlántico. Deliberadamente lejos de cualquier tile
    # real (no un punto de mar cercano a España, que podría caer dentro de
    # un tile existente y devolver 0 en vez de lanzar la excepción).
    with pytest.raises(TileNotFoundError):
        elevation_m(0.0, -30.0, dem_dir=DEM_DIR)


@pytest.mark.skipif(
    not PENALARA_TILE.exists(),
    reason=f"falta {PENALARA_TILE} (proyecto movido a Málaga, no se ha "
    "descargado cobertura de Sierra de Guadarrama)",
)
def test_altitud_penalara():
    assert 2400 <= elevation_m(PENALARA[0], PENALARA[1], dem_dir=DEM_DIR) <= 2430


def test_altitud_maroma():
    # La Maroma (Wikipedia): 36.90194, -4.04444, altitud oficial 2069 m.
    # Cae en data/N36W005.hgt. Tolerancia amplia a propósito: la rejilla de
    # 30 m rara vez cae en la cima exacta y las fuentes discrepan ~100 m en
    # la posición. Aun así detecta bugs reales: orden de bytes invertido da
    # valores absurdos, filas invertidas da ~500-900 m, tile equivocado da
    # otra altitud completamente distinta.
    alt = elevation_m(36.90194, -4.04444, dem_dir=DEM_DIR)
    assert 2000 <= alt <= 2075


def test_altitud_mar_abierto():
    # (36.65, -4.10): mar abierto cerca de Málaga, dentro de N36W005.hgt.
    # Comprobado en ejecución real: este DEM da 0.0 (no None/void) para el
    # mar. SRTM no es consistente entre tiles sobre esto -- otros tiles
    # codifican el mar como void (-32768). Si este test empieza a fallar
    # con None en vez de 0, significa que se ha cambiado de tile/fuente y
    # hay que revisar el comentario, no ajustar el valor a ciegas.
    assert elevation_m(36.65, -4.10, dem_dir=DEM_DIR) == pytest.approx(0.0, abs=1.0)


# --- src/horizon/ -------------------------------------------------------


def test_destino_sentido():
    # Caso 4 del contrato, en su lectura literal ("avanzar con azimut 0°").
    lat, lon = destination_point_deg(36.730, -4.097, 0.0, 10_000.0)
    assert lat > 36.730
    assert lon == pytest.approx(-4.097, abs=1e-9)
    lat, lon = destination_point_deg(36.730, -4.097, 90.0, 10_000.0)
    assert lon > -4.097
    assert lat == pytest.approx(36.730, abs=0.01)


def test_destino_ida_y_vuelta():
    # El directo y el inverso deben cerrar el círculo.
    lat, lon = destination_point_deg(36.730, -4.097, 42.0, 25_000.0)
    assert haversine_m(36.730, -4.097, lat, lon) == pytest.approx(25_000.0, rel=1e-6)
    assert azimuth_deg(36.730, -4.097, lat, lon) == pytest.approx(42.0, abs=1e-6)


def test_version_vectorizada_coincide_con_escalar():
    # El barrido usa las versiones numpy; si divergieran de las escalares
    # (que son las que validan los tests dorados de geo/) el error sería
    # invisible. Este test ata las dos implementaciones.
    distances_m = np.array([30.0, 1_000.0, 12_090.0, 50_000.0, 150_000.0])
    for bearing_deg in (0.0, 45.0, 90.0, 180.0, 270.0, 336.0):
        lat_arr, lon_arr = destination_points_deg(36.730, -4.097, bearing_deg,
                                                  distances_m)
        for i, d_m in enumerate(distances_m):
            lat, lon = destination_point_deg(36.730, -4.097, bearing_deg, d_m)
            assert lat == pytest.approx(lat_arr[i], abs=1e-12)
            assert lon == pytest.approx(lon_arr[i], abs=1e-12)

    heights_m = np.array([0.0, 500.0, 2_000.0, -10.0, 0.0])
    vector = elevations_deg(distances_m, 10.0, heights_m)
    for i, d_m in enumerate(distances_m):
        assert elevation_deg(d_m, 10.0, heights_m[i]) == pytest.approx(
            vector[i], abs=1e-12)


def test_bilineal_vectorizada_coincide_con_escalar():
    # check_visibility usa elevation_m (escalar) y horizon_profile usa
    # elevations_m (vectorizada): la bilineal está escrita dos veces. Si
    # divergieran, las dos funciones públicas se contradirían sin fallar.
    lat_arr, lon_arr = destination_points_deg(MAROMA[0], MAROMA[1], 200.0,
                                              np.arange(30.0, 20_000.0, 30.0))
    vector = elevations_m(lat_arr, lon_arr, dem_dir=DEM_DIR)
    # elevation_m marca los voids con None y elevations_m con NaN. Sin
    # convertir el None el array saldría dtype object, donde equal_nan ni
    # siquiera aplica.
    scalar = np.array([np.nan if v is None else v
                       for v in (elevation_m(la, lo, dem_dir=DEM_DIR)
                                 for la, lo in zip(lat_arr, lon_arr))],
                      dtype=float)
    # equal_nan porque NaN nunca es igual a sí mismo: sin esto el test
    # fallaría espuriamente en cuanto el rayo cruzase un void.
    assert np.array_equal(vector, scalar, equal_nan=True)


def test_horizonte_marino():
    # Desde 10 m sobre el mar mirando al sur solo hay agua (altitud 0). El
    # máximo tiene solución cerrada: maximizando f(d) = (-h - 0.87 d²/(2R))/d
    # sale d = sqrt(2Rh/0.87) = 12102.1 m y el ángulo -0.094688°.
    # Valida de golpe el signo de la curvatura (si estuviera cambiado saldría
    # positivo), el convenio de elevaciones negativas, la regla del máximo
    # acumulado y la fórmula de destino. Sin curvatura daría -0.0473°.
    prof = horizon_profile(MAR[0], MAR[1], 10.0, dem_dir=DEM_DIR)
    idx = int(np.flatnonzero(prof.azimuths_deg == 180.0)[0])
    assert prof.elevations_deg[idx] == pytest.approx(-0.0947, abs=0.005)

    # El rayo al sur se queda sin cobertura antes de los 150 km (pide
    # N35W005.hgt). Lo que importa es QUE se truncó, no dónde: la distancia
    # exacta depende de qué tiles haya en data/ y cambiaría al descargar uno
    # más. El máximo se alcanza a ~12 km, mucho antes del corte, así que el
    # valor del perfil sigue siendo válido.
    assert not np.isnan(prof.truncated_at_m[idx])


def test_check_visibility_tres_estados():
    # Todo desde el mismo punto de mar mirando al sur, con datos disponibles.
    lat_obs, lon_obs, h_obs_m = MAR[0], MAR[1], 10.0

    # A 20 km y a nivel del mar: lo tapa el propio horizonte marino, que está
    # a -0.0947° mientras el objetivo queda a -0.107°.
    lat, lon = destination_point_deg(lat_obs, lon_obs, 180.0, 20_000.0)
    blocked = check_visibility(lat_obs, lon_obs, h_obs_m, lat, lon, 0.0,
                               dem_dir=DEM_DIR)
    assert blocked.status is Visibility.BLOCKED

    # Mismo punto elevado a 500 m: queda muy por encima del horizonte.
    visible = check_visibility(lat_obs, lon_obs, h_obs_m, lat, lon, 500.0,
                               dem_dir=DEM_DIR)
    assert visible.status is Visibility.VISIBLE

    # A 100 km: nada lo bloquea antes, pero el rayo se sale de la cobertura.
    # No puede responderse, y eso se dice explícitamente en vez de devolver
    # un VISIBLE que sería mentira.
    lat, lon = destination_point_deg(lat_obs, lon_obs, 180.0, 100_000.0)
    unknown = check_visibility(lat_obs, lon_obs, h_obs_m, lat, lon, 2_000.0,
                               dem_dir=DEM_DIR)
    assert unknown.status is Visibility.UNKNOWN
    # Igual que arriba: la distancia concreta depende de la cobertura de
    # data/, solo se comprueba que se registró el truncamiento.
    assert unknown.truncated_at_m is not None
    assert unknown.truncated_at_m > 0.0


def test_visibilidad_maroma_al_mar():
    # Equivalente al caso 6 del contrato que sí corre con los tiles de
    # Andalucía. Desde la cima de La Maroma (ojo = cota DEM + 1.7 m) hasta un
    # punto de mar a ~19.7 km: nada puede taparlo desde 2043 m, y el objetivo
    # queda a -6.0° (muy por debajo del observador).
    h_eye_m = elevation_m(MAROMA[0], MAROMA[1], dem_dir=DEM_DIR) + 1.7
    result = check_visibility(MAROMA[0], MAROMA[1], h_eye_m,
                              MAR[0], MAR[1], 0.0, dem_dir=DEM_DIR)
    assert result.status is Visibility.VISIBLE


@pytest.mark.skipif(
    not PENALARA_TILE.exists(),
    reason=f"falta {PENALARA_TILE} (proyecto movido a Málaga, no se ha "
    "descargado cobertura de Sierra de Guadarrama)",
)
def test_visibilidad_penalara_bola_del_mundo():
    # Caso 6 del contrato: ~7 km sin obstáculos entre ambas cimas.
    h_eye_m = PENALARA[2] + 1.7
    result = check_visibility(PENALARA[0], PENALARA[1], h_eye_m,
                              BOLA_DEL_MUNDO[0], BOLA_DEL_MUNDO[1],
                              BOLA_DEL_MUNDO[2], dem_dir=DEM_DIR)
    assert result.status is Visibility.VISIBLE


def test_truncamiento_por_distancia_no_por_tile():
    # Desde (36.97, -1.06) con azimut 60° el rayo sale de N36W002 y entra en
    # N36W001 (que NO está descargado), pero más adelante vuelve a pisar
    # N37W001, que sí lo está. El corte tiene que ser la primera muestra
    # ausente EN ORDEN DE DISTANCIA, descartando todo lo posterior aunque
    # haya cobertura: agrupar por identidad de tile podría cortar más tarde.
    lat_obs, lon_obs, bearing_deg, max_km = 36.97, -1.06, 60.0, 20.0

    # El corte esperado se recalcula aquí en vez de fijar un número: la
    # distancia concreta depende de qué tiles haya en data/.
    distances_m = np.arange(30.0, max_km * 1000.0 + 15.0, 30.0)
    lat_arr, lon_arr = destination_points_deg(lat_obs, lon_obs, bearing_deg,
                                              distances_m)
    available = np.array([(Path(DEM_DIR) / locate_tile_path(la, lo)).exists()
                          for la, lo in zip(lat_arr, lon_arr)])
    cut = int(np.flatnonzero(~available)[0])
    # el caso solo es interesante si después del corte queda cobertura
    assert available[cut + 1:].any()

    prof = horizon_profile(lat_obs, lon_obs, 10.0, max_km=max_km, dem_dir=DEM_DIR)
    idx = int(np.argmin(np.abs(prof.azimuths_deg - bearing_deg)))
    assert prof.truncated_at_m[idx] == pytest.approx(distances_m[cut])


@pytest.fixture
def tile_sintetico_con_void(tmp_path):
    """Tile SRTM3 (1201x1201) enteramente void salvo un parche de terreno al
    norte del centro.

    Los voids reales de data/ están todos en la columna 3600 (borde este) de
    los tiles W001/W002 y son inalcanzables vía elevation_m: floor() enruta
    esa longitud exacta al tile siguiente al este, que no está descargado, y
    sale TileNotFoundError en vez de void. Por eso el void hay que fabricarlo.

    De paso ejerce el camino SRTM3 de load_tile, que los tiles reales
    (todos SRTM1) no tocan.
    """
    n = 1200
    array = np.full((n + 1, n + 1), VOID, dtype=">i2")
    # lat 10.52..10.54, lon -9.52..-9.48 -> terreno a 1000 m
    array[552:577, 576:625] = 1000
    path = tmp_path / "N10W010.hgt"
    array.tofile(path)
    assert path.stat().st_size == 2_884_802  # tamaño exacto SRTM3
    return str(tmp_path)


def test_rayo_con_void_no_envenena_el_maximo(tile_sintetico_con_void):
    # Con np.max en vez de np.nanmax, un solo void haría NaN el azimut entero.
    prof = horizon_profile(10.5, -9.5, 100.0, max_km=5.0,
                           dem_dir=tile_sintetico_con_void)

    # Al norte el rayo cruza el parche de terreno entre muchas muestras void:
    # las válidas deben seguir contando.
    idx_norte = int(np.flatnonzero(prof.azimuths_deg == 0.0)[0])
    assert not np.isnan(prof.elevations_deg[idx_norte])
    assert prof.elevations_deg[idx_norte] > 0.0

    # Al sur el rayo es íntegramente void: el perfil es NaN, no un número
    # inventado ni el -32768 disfrazado.
    idx_sur = int(np.flatnonzero(prof.azimuths_deg == 180.0)[0])
    assert np.isnan(prof.elevations_deg[idx_sur])

    # Nada se truncó: los 5 km caben de sobra dentro del tile sintético.
    assert np.all(np.isnan(prof.truncated_at_m))


# --- validación externa contra PeakFinder -------------------------------
#
# tests/data/peakfinder_referencia.csv son 72 cimas con azimut y distancia
# producidos por PeakFinder, una implementación INDEPENDIENTE que no comparte
# nuestro código ni nuestros errores.
#
# A diferencia del resto del fichero, estos dos tests no validan una función
# aislada ni un caso de solución cerrada que hayamos calculado nosotros
# mismos: validan el motor COMPLETO de punta a punta — geo (azimut y
# distancia) + dem (lectura e interpolación del terreno) + horizon (barrido de
# 360° y regla del máximo acumulado) — contra un tercero. Es la única
# comprobación del proyecto que puede detectar un error de convenio compartido
# por todos nuestros módulos.
#
# El CSV se versiona porque es texto (~4 KB), no un tile: .gitignore excluye
# data/*.hgt pero no toca tests/data/.


@pytest.fixture(scope="module")
def peakfinder_rows() -> list[dict]:
    with open(PEAKFINDER_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 72
    return rows


@pytest.fixture(scope="module")
def peakfinder_profile():
    # scope="module": el barrido de 1800 rayos tarda ~4 s y lo comparten los
    # dos tests en vez de calcularse dos veces.
    h_eye_m = elevation_m(*PEAKFINDER_OBS, dem_dir=DEM_DIR) + 1.7
    return h_eye_m, horizon_profile(PEAKFINDER_OBS[0], PEAKFINDER_OBS[1],
                                    h_eye_m, dem_dir=DEM_DIR)


def test_peakfinder_azimuts(peakfinder_rows):
    # Medido: peor caso 0.085°, RMS 0.040° — holgura de 6x sobre el umbral.
    failures = []
    for row in peakfinder_rows:
        computed_deg = azimuth_deg(PEAKFINDER_OBS[0], PEAKFINDER_OBS[1],
                                   float(row["lat"]), float(row["lon"]))
        reference_deg = float(row["azimut_deg"])
        # aritmética modular: sin esto una cima al norte con azimut 359.9 vs
        # 0.1 daría un falso fallo de 359.8°
        delta_deg = abs((computed_deg - reference_deg + 180.0) % 360.0 - 180.0)
        if delta_deg >= 0.5:
            failures.append(f"{row['nombre']}: {delta_deg:.3f}° "
                            f"(calculado {computed_deg:.2f}, "
                            f"PeakFinder {reference_deg:.1f})")
    # se acumulan todos los fallos en vez de abortar en el primero: que falle
    # una cima o las 72 tiene diagnósticos opuestos (ver CLAUDE.md)
    assert not failures, (f"{len(failures)}/72 cimas fuera de 0.5°:\n  "
                          + "\n  ".join(failures))


def test_peakfinder_perfil_alcanza_las_cimas(peakfinder_rows, peakfinder_profile):
    # El perfil del horizonte en el azimut de cada cima debe llegar al menos a
    # su elevación menos 0.2°.
    #
    # El criterio es UNILATERAL a propósito, porque los dos signos del margen
    # (perfil - requerido) significan cosas distintas. Medido sobre las 72:
    # 41 negativos y 31 POSITIVOS, media +0.196°, mediana -0.006°.
    #
    #   Margen positivo, por grande que sea (el mayor medido es +2.517°): es
    #   NORMAL, no un bug, y no tiene cota superior esperable. El perfil es el
    #   máximo de todo el rayo mientras que el requerido es la elevación de
    #   UNA cima; una cima de primer plano puede tener detrás otra más alta
    #   casi en el mismo azimut, que domina el perfil. Comprobado: a Cerro del
    #   Tablón (824 m, 14.6 km) lo tapa Cerro la Cuna (1630 m, 17.6 km), y a
    #   Cerro Juan María (576 m) lo tapa Cerro Tacita de Plata (1893 m).
    #
    #   Margen negativo: acotado a ~0.1° (el peor medido es -0.072°, en
    #   Benthomiz) por el sesgo conocido de SRTM, que subestima las cimas por
    #   promediado. Si algún día aparece un déficit negativo GRANDE, más allá
    #   del -0.2° de tolerancia, NO es ese sesgo: es un bug distinto — el rayo
    #   no llega a la cima, el azimut está desplazado, o el DEM se lee mal.
    #
    # Es decir, lo sospechoso es el déficit negativo grande, nunca el exceso
    # positivo.
    h_eye_m, profile = peakfinder_profile
    failures = []
    for row in peakfinder_rows:
        lat_deg, lon_deg = float(row["lat"]), float(row["lon"])
        # ele_m viene en metros y dist_km en kilómetros: es el único sitio del
        # proyecto donde entran km, y solo porque el fichero externo es así
        d_m = haversine_m(PEAKFINDER_OBS[0], PEAKFINDER_OBS[1], lat_deg, lon_deg)
        required_deg = elevation_deg(d_m, h_eye_m, float(row["ele_m"]))
        bearing_deg = azimuth_deg(PEAKFINDER_OBS[0], PEAKFINDER_OBS[1],
                                  lat_deg, lon_deg)
        i = int(np.argmin(np.abs(profile.azimuths_deg - bearing_deg)))
        margin_deg = profile.elevations_deg[i] - required_deg
        if not (margin_deg >= -0.2):  # cubre también el NaN del perfil
            failures.append(f"{row['nombre']}: margen {margin_deg:+.3f}° "
                            f"(perfil {profile.elevations_deg[i]:.3f}, "
                            f"requerido {required_deg:.3f})")
    assert not failures, (f"{len(failures)}/72 cimas por debajo del perfil:\n  "
                          + "\n  ".join(failures))


# --- src/peaks/ ---------------------------------------------------------


def test_parse_ele():
    assert _parse_ele("2069") == 2069.0
    assert _parse_ele("2069 m") == 2069.0
    assert _parse_ele(" 1234.5 M ") == 1234.5
    assert _parse_ele("unknown") is None
    assert _parse_ele(None) is None


def test_fetch_peaks_desde_cache(tmp_path, monkeypatch):
    # El nombre del fichero de caché es determinista sobre los parámetros
    # redondeados: escribir el fixture con ese nombre equivale a una
    # respuesta ya cacheada. La red se bloquea para DEMOSTRAR que no se toca.
    import src.peaks as peaks_module
    monkeypatch.setattr(peaks_module.requests, "post",
                        lambda *a, **k: pytest.fail("fetch_peaks tocó la red "
                                                    "con caché presente"))
    payload = {"elements": [
        {"type": "node", "id": 1, "lat": 36.9, "lon": -4.0,
         "tags": {"name": "Pico A", "ele": "1500 m", "wikidata": "Q1"}},
        {"type": "node", "id": 2, "lat": 36.8, "lon": -4.1,
         "tags": {"ele": "basura"}},           # sin nombre, ele no parseable
        {"type": "way", "id": 3},              # no es un nodo: se ignora
    ]}
    cache_file = tmp_path / "overpass" / "peaks_36.7450_-4.0900_r100000.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    peaks = fetch_peaks(36.745, -4.090, 100_000, cache_dir=str(tmp_path))
    assert len(peaks) == 2
    assert peaks[0].name == "Pico A"
    assert peaks[0].ele_m == 1500.0
    assert peaks[0].wikidata == "Q1"
    assert peaks[1].name is None
    assert peaks[1].ele_m is None  # "basura" se trata como ausente


@pytest.fixture
def tile_sintetico_con_cima(tmp_path):
    """Tile SRTM3 con base 500 m y un único nodo a 1500 m en (10.5, -9.5)
    (fila 600, columna 600: lat = 10 + 1 - 600/1200, lon = -10 + 600/1200)."""
    array = np.full((1201, 1201), 500, dtype=">i2")
    array[600, 600] = 1500
    array.tofile(tmp_path / "N10W010.hgt")
    return str(tmp_path)


def test_relocate_peak(tile_sintetico_con_cima):
    cima = (10.5, -9.5)
    # Coordenada OSM "puesta a ojo" 100 m al norte de la cima real: la
    # recolocación debe encontrar el nodo alto (rejilla de 15 m: error
    # residual máximo ~11 m).
    desviado = Peak(1, "X", cima[0] + 100.0 / 111_195.0, cima[1], None, None)
    lat_r, lon_r = relocate_peak(desviado, dem_dir=tile_sintetico_con_cima)
    assert haversine_m(lat_r, lon_r, *cima) < 20.0

    # Ya está en el máximo: no moverse (empate = quedarse).
    exacto = Peak(2, "X", *cima, None, None)
    assert relocate_peak(exacto, dem_dir=tile_sintetico_con_cima) == cima

    # Fuera de cobertura (N10W009 no existe): coordenada original, sin
    # excepción — la recolocación es una corrección, no un requisito.
    fuera = Peak(3, "X", 10.5, -8.5, None, None)
    assert relocate_peak(fuera, dem_dir=tile_sintetico_con_cima) == (10.5, -8.5)


def test_ray_visibility_coincide_con_check_visibility():
    # _ray_visibility (vectorizada, para evaluar cientos de picos) y
    # check_visibility (escalar) son la misma semántica escrita dos veces:
    # este test las ata, como los demás pares escalar/vectorizado del
    # proyecto. Se comparan sobre los tres estados.
    lat_obs, lon_obs, h_obs_m = MAR[0], MAR[1], 10.0
    casos = [(20_000.0, 0.0), (20_000.0, 500.0), (100_000.0, 2_000.0)]
    for d_km_m, h_tgt_m in casos:
        lat_t, lon_t = destination_point_deg(lat_obs, lon_obs, 180.0, d_km_m)
        escalar = check_visibility(lat_obs, lon_obs, h_obs_m,
                                   lat_t, lon_t, h_tgt_m, dem_dir=DEM_DIR)
        d_m = haversine_m(lat_obs, lon_obs, lat_t, lon_t)
        bearing_deg = azimuth_deg(lat_obs, lon_obs, lat_t, lon_t)
        target_deg = elevation_deg(d_m, h_obs_m, h_tgt_m)
        vectorial = _ray_visibility(lat_obs, lon_obs, h_obs_m, bearing_deg,
                                    d_m, target_deg, DEM_DIR, {})
        assert vectorial.status is escalar.status
        for campo in ("truncated_at_m", "blocked_at_m"):
            esc, vec = getattr(escalar, campo), getattr(vectorial, campo)
            if esc is None:
                assert vec is None
            else:
                assert vec == pytest.approx(esc, abs=STEP_TOL_M)


STEP_TOL_M = 30.0  # una muestra de diferencia entre implementaciones es ruido

# Altura del ojo para reproducir el panorama de PeakFinder. El CSV no la
# registra, así que se deriva de los datos, igual que el observador:
# PeakFinder eleva la cámara virtual sobre el terreno para evitar el
# desorden del primer plano. Medido con este motor (cimas no-BLOCKED de 72):
#   ojo  4.7 m (cota DEM + 1.7) -> 62    ojo 13-20 m -> 67 (meseta estable)
#   ojo 10.0 m                  -> 66
# 15 m es el centro de la meseta: el resultado no depende del valor exacto.
PEAKFINDER_EYE_M = 15.0


def test_peakfinder_visibilidad(peakfinder_rows):
    # Validación externa del pipeline completo de peaks/ (recolocación +
    # altitud OSM + visibilidad por rayo) contra el panorama de PeakFinder:
    # si PeakFinder muestra una cima en su silueta, nuestro motor debería
    # verla también (VISIBLE) o quedarse sin datos para negarlo (UNKNOWN).
    #
    # El umbral es 90% y no 100% porque en los casos marginales (crestas que
    # rozan la línea de visión con excesos de centésimas de grado) la
    # diferencia entre DEMs y muestreos decide el resultado; 5 de las 72
    # están en esa zona gris con cualquier altura de ojo razonable.
    peaks = [Peak(osm_id=i, name=row["nombre"], lat_deg=float(row["lat"]),
                  lon_deg=float(row["lon"]), ele_m=float(row["ele_m"]),
                  wikidata=None)
             for i, row in enumerate(peakfinder_rows)]
    sightings = evaluate_peaks(peaks, PEAKFINDER_OBS[0], PEAKFINDER_OBS[1],
                               PEAKFINDER_EYE_M, dem_dir=DEM_DIR)

    seen = {s.peak.name for s in sightings}
    blocked = [p.name for p in peaks if p.name not in seen]
    assert len(sightings) >= math.ceil(0.9 * 72), (
        f"solo {len(sightings)}/72 salen VISIBLE/UNKNOWN; "
        f"bloqueadas: {blocked}")

    # Las cuatro dominantes son el techo del panorama: nada puede taparlas.
    # Si una sale BLOCKED es un bug (p. ej. auto-bloqueo del rayo con la
    # ladera del propio pico), no un caso marginal.
    status = {s.peak.name: s.visibility for s in sightings}
    for name in ("Maroma", "Cima de Tejeda", "Cerro del Sol",
                 "Mojón de tres Términos"):
        assert status.get(name) is Visibility.VISIBLE, (
            f"{name} debería ser VISIBLE y es "
            f"{status.get(name, 'BLOCKED')}")


# --- CLI (python -m peakid) ---------------------------------------------


def test_cli_photo_reservado():
    from peakid.__main__ import main
    # --photo está reservado para la fase 2: debe salir con código 4 ANTES
    # de tocar DEM o red (por eso funciona sin fixtures).
    code = main(["panorama", "--lat", "36.745", "--lon", "-4.090",
                 "--photo", "foto.jpg"])
    assert code == 4


def test_cli_lon_negativa_es_valida():
    from peakid.__main__ import main
    # La razón de --lat/--lon con nombre: una longitud oeste como argumento
    # posicional rompería argparse. Con nombre debe llegar al handler (y
    # fallar por otra cosa controlada, no por el parseo).
    code = main(["visible", "--lat", "36.745", "--lon", "-4.090",
                 "--target-lat", "36.902", "--target-lon", "-4.044",
                 "--target-ele", "2069"])
    assert code == 0


def test_cli_visible_blocked(capsys):
    from peakid.__main__ import main
    # El caso BLOCKED conocido: objetivo a nivel del mar a 20 km, tapado por
    # el horizonte marino (~12 km). Debe imprimir dónde está el obstáculo.
    lat_t, lon_t = destination_point_deg(MAR[0], MAR[1], 180.0, 20_000.0)
    code = main(["visible", "--lat", str(MAR[0]), "--lon", str(MAR[1]),
                 "--eye", "10.0", "--target-lat", f"{lat_t:.6f}",
                 "--target-lon", f"{lon_t:.6f}", "--target-ele", "0"])
    out = capsys.readouterr().out
    assert code == 0
    assert "BLOCKED" in out
    assert "obstáculo a " in out
    # El obstáculo es la PRIMERA muestra que corta la línea de visión, no el
    # máximo del horizonte marino (que está a 12.1 km): la visual al objetivo
    # (-0.107°) se hunde bajo el mar donde -h/d - 0.87·d/(2R) = -0.107°, cuya
    # solución cerrada con h=10 es d = 7302 m.
    km = float(out.split("obstáculo a ")[1].split(" km")[0])
    assert km == pytest.approx(7.3, abs=0.2)


def test_cli_tile_ausente_sin_traceback(capsys):
    from peakid.__main__ import main
    code = main(["visible", "--lat", "0.0", "--lon", "-30.0",
                 "--target-lat", "0.5", "--target-lon", "-30.0",
                 "--target-ele", "100"])
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert "S01W030.hgt" in captured.err or "N00W030.hgt" in captured.err
    assert "Traceback" not in captured.err


def test_cli_peaks_csv(tmp_path):
    from peakid.__main__ import main
    # Overpass simulado vía caché (mismo mecanismo que test_fetch_peaks):
    # un nodo tipo Maroma que debe salir VISIBLE en el CSV.
    payload = {"elements": [
        {"type": "node", "id": 1, "lat": MAROMA[0], "lon": MAROMA[1],
         "tags": {"name": "Maroma", "ele": "2069"}},
    ]}
    cache_file = tmp_path / "overpass" / "peaks_36.7450_-4.0900_r100000.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    out_csv = tmp_path / "picos.csv"

    code = main(["peaks", "--lat", "36.745", "--lon", "-4.090",
                 "--eye", "15.0", "--cache-dir", str(tmp_path),
                 "--csv", str(out_csv)])
    assert code == 0
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["nombre"] == "Maroma"
    assert rows[0]["estado"] == "visible"
    assert rows[0]["fuente_altitud"] == "osm"
    assert float(rows[0]["dist_km"]) == pytest.approx(18.0, abs=0.5)
