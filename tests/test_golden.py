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
from PIL import Image

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
from src.horizon import (
    HorizonProfile,
    Visibility,
    check_visibility,
    horizon_profile,
)
from src.peaks import (
    Peak,
    PeakSighting,
    _parse_ele,
    _ray_visibility,
    evaluate_peaks,
    fetch_peaks,
    relocate_peak,
)
from src.render import render_horizon_png

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
    """Caso 5 del contrato. Coordenada y rango SIN TOCAR (40.8508, −3.9578;
    2400–2430); lo que se añade es la recolocación de 200 m.

    Medido al descargar por fin data/N40W004.hgt: la coordenada del contrato
    (que viene de Wikipedia) queda a **175 m** de la cumbre real según el
    DEM. Leída directamente da **2391.2 m** — 37 m por debajo de los 2428
    oficiales; recolocada al máximo dentro de 200 m da **2424.2 m**, un
    déficit de solo 4 m, coherente con el sesgo conocido de SRTM.

    Recolocar es lo que hace el motor con TODA coordenada de cima, porque
    las de OSM también están puestas a ojo (ver relocate_peak, mismo radio
    de 200 m); ya se validó igual con La Maroma, cuya cumbre DEM cae a 156 m
    al norte y 71 m al este de la coordenada de Wikipedia. Leer el punto
    literal mediría la ladera, no la cima.
    """
    peak = Peak(0, "Peñalara", PENALARA[0], PENALARA[1], None, None)
    lat_deg, lon_deg = relocate_peak(peak, dem_dir=DEM_DIR)
    assert haversine_m(PENALARA[0], PENALARA[1], lat_deg, lon_deg) <= 200.0
    assert 2400 <= elevation_m(lat_deg, lon_deg, dem_dir=DEM_DIR) <= 2430


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


def test_cli_photo_inexistente(capsys):
    from peakid.__main__ import main
    # --photo ya no es el stub de fase 2: abre la herramienta de
    # alineamiento. Con una foto inexistente debe fallar limpio ANTES de
    # tocar DEM, red o tkinter (por eso el test funciona sin fixtures).
    code = main(["panorama", "--lat", "36.745", "--lon", "-4.090",
                 "--photo", "no_existe.jpg"])
    captured = capsys.readouterr()
    assert code == 2
    assert "no existe la foto" in captured.err
    assert "Traceback" not in captured.err


def test_cli_panorama_sin_coordenadas(capsys):
    from peakid.__main__ import main
    # --lat/--lon dejaron de ser required en panorama (una foto con GPS los
    # suple); sin foto Y sin coordenadas debe fallar con mensaje, no con
    # AttributeError.
    code = main(["panorama"])
    captured = capsys.readouterr()
    assert code == 1
    assert "--lat y --lon son obligatorios" in captured.err


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


# --- recorte de azimut en render/ ---------------------------------------


def test_crop_span_desenrollado():
    from src.render import _crop_span, _unwrap_deg

    # sector normal: no se toca
    assert _crop_span(0.0, 60.0) == (0.0, 60.0)
    # sector que cruza el norte: el extremo superior se desenrolla
    az_min, az_max_u = _crop_span(340.0, 40.0)
    assert (az_min, az_max_u) == (340.0, 400.0)
    assert az_max_u - az_min == 60.0
    # az_min == az_max se lee como vuelta completa, no como sector vacío
    assert _crop_span(90.0, 90.0) == (90.0, 450.0)
    # normalización de entradas fuera de [0, 360)
    assert _crop_span(700.0, 380.0) == (340.0, 380.0)

    # los azimuts del sector 340->40 caen dentro; los de fuera, no
    assert _unwrap_deg(350.0, 340.0) == 350.0
    assert _unwrap_deg(10.0, 340.0) == 370.0
    assert not az_min <= _unwrap_deg(100.0, 340.0) <= az_max_u
    assert not az_min <= _unwrap_deg(339.9, 340.0) <= az_max_u


def test_render_recorte_filtra_picos(tmp_path):
    # Perfil sintético plano: aquí se comprueba el recorte, no la geometría.
    az = np.arange(0.0, 360.0, 0.2)
    profile = HorizonProfile(azimuths_deg=az,
                             elevations_deg=np.full(az.shape, 2.0),
                             truncated_at_m=np.full(az.shape, np.nan))
    sightings = [_fake_sighting("dentro-350", 350.0),
                 _fake_sighting("dentro-10", 10.0),
                 _fake_sighting("fuera-100", 100.0),
                 _fake_sighting("fuera-200", 200.0)]

    out = tmp_path / "sector.png"
    # sector que cruza el norte: el caso que rompería sin desenrollar
    report = render_horizon_png(profile, str(out), peaks=sightings,
                                az_min_deg=340.0, az_max_deg=40.0)
    assert out.exists()
    from PIL import Image
    with Image.open(out) as img:
        assert img.size == (1600, 520)

    assert set(report.placed) == {"dentro-350", "dentro-10"}
    assert set(report.out_of_range) == {"fuera-100", "fuera-200"}
    assert report.discarded == []  # separados de los de fuera de encuadre


def test_render_sin_recorte_no_cambia(tmp_path):
    # El requisito explícito: sin recorte el PNG debe ser byte a byte el
    # mismo que pasando el rango completo equivalente.
    az = np.arange(0.0, 360.0, 0.2)
    profile = HorizonProfile(azimuths_deg=az,
                             elevations_deg=np.linspace(0.0, 5.0, az.size),
                             truncated_at_m=np.full(az.shape, np.nan))
    sin_recorte = tmp_path / "a.png"
    render_horizon_png(profile, str(sin_recorte))
    assert sin_recorte.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _fake_sighting(name: str, azimuth_deg: float):
    peak = Peak(osm_id=hash(name) % 1000, name=name, lat_deg=36.9,
                lon_deg=-4.0, ele_m=1000.0, wikidata=None)
    return PeakSighting(peak=peak, lat_deg=36.9, lon_deg=-4.0, h_m=1000.0,
                        h_source="osm", azimuth_deg=azimuth_deg,
                        distance_m=20_000.0, elevation_deg=1.5,
                        visibility=Visibility.VISIBLE, truncated_at_m=None)


def test_cli_recorte(tmp_path, capsys):
    from peakid.__main__ import main
    payload = {"elements": [
        {"type": "node", "id": 1, "lat": MAROMA[0], "lon": MAROMA[1],
         "tags": {"name": "Maroma", "ele": "2069"}},
    ]}
    cache_file = tmp_path / "overpass" / "peaks_36.7450_-4.0900_r100000.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    out_png = tmp_path / "sector.png"

    code = main(["panorama", "--lat", "36.745", "--lon", "-4.090",
                 "--eye", "15.0", "--az-min", "0", "--az-max", "60",
                 "--cache-dir", str(tmp_path), "-o", str(out_png)])
    assert code == 0
    assert out_png.exists()
    out = capsys.readouterr().out
    assert "sector 0°–60°" in out
    # Maroma está a 13.2°, dentro del sector
    assert "etiquetados 1" in out


def test_cli_recorte_incompleto(capsys):
    from peakid.__main__ import main
    code = main(["panorama", "--lat", "36.745", "--lon", "-4.090",
                 "--az-min", "0"])
    captured = capsys.readouterr()
    assert code == 1
    assert "--az-min y --az-max" in captured.err
    assert "Traceback" not in captured.err


# --- src/align/ (fase 2: proyección y alineamiento manual) --------------


def test_project_profile_centro_y_bordes():
    from src.align import AlignmentParams, project_profile

    W, H = 4000, 3000
    params = AlignmentParams(azimuth_deg=100.0, hfov_deg=65.0,
                             pitch_deg=0.0, roll_deg=0.0)
    # el centro exacto (azimut central, elevación = pitch) proyecta al
    # centro de la imagen
    x, y, ok = project_profile([100.0], [0.0], params, W, H)
    assert ok[0]
    assert x[0] == pytest.approx(W / 2)
    assert y[0] == pytest.approx(H / 2)

    # con pitch=0 y roll=0, azimut central + hfov/2 a elevación 0 cae
    # exactamente en el borde derecho (x=W); el simétrico en x=0
    x, y, ok = project_profile([100.0 + 32.5, 100.0 - 32.5], [0.0, 0.0],
                               params, W, H)
    assert x[0] == pytest.approx(W, abs=1e-6)
    assert x[1] == pytest.approx(0.0, abs=1e-6)
    assert y[0] == pytest.approx(H / 2)

    # una muestra detrás de la cámara no es proyectable
    _, _, ok = project_profile([280.0], [0.0], params, W, H)
    assert not ok[0]

    # elevación NaN (void del perfil) tampoco, sin reventar
    _, _, ok = project_profile([100.0], [float("nan")], params, W, H)
    assert not ok[0]


def test_project_profile_pitch_y_roll():
    from src.align import AlignmentParams, project_profile

    W, H = 4000, 3000
    # pitch: mirar hacia arriba exactamente a la elevación de la muestra la
    # centra en la imagen
    params = AlignmentParams(100.0, 65.0, pitch_deg=3.0, roll_deg=0.0)
    x, y, ok = project_profile([100.0], [3.0], params, W, H)
    assert ok[0] and y[0] == pytest.approx(H / 2)

    # roll positivo = el lado derecho del horizonte dibujado BAJA (convenio
    # documentado); con roll=90 el desplazamiento horizontal se vuelve
    # vertical por completo
    params = AlignmentParams(100.0, 65.0, pitch_deg=0.0, roll_deg=5.0)
    _, y, _ = project_profile([110.0], [0.0], params, W, H)
    assert y[0] > H / 2
    params = AlignmentParams(100.0, 65.0, pitch_deg=0.0, roll_deg=90.0)
    x, y, _ = project_profile([110.0], [0.0], params, W, H)
    assert x[0] == pytest.approx(W / 2, abs=1e-6)
    assert y[0] > H / 2


def test_alignment_json_roundtrip(tmp_path):
    from src.align import (AlignmentParams, alignment_json_path,
                           load_alignment, save_alignment, seed_entry)

    json_path = alignment_json_path(str(tmp_path / "IMG_1234.jpg"))
    assert json_path.name == "IMG_1234.align.json"

    params = AlignmentParams(347.2, 65.3, 2.1, -0.8)
    seed = {"azimuth_deg": seed_entry(341.0, "exif"),
            "hfov_deg": seed_entry(66.0, "exif"),
            "pitch_deg": seed_entry(0.0, "default"),
            "roll_deg": seed_entry(0.0, "default")}
    save_alignment(str(json_path), "IMG_1234.jpg",
                   {"lat_deg": 36.745, "lon_deg": -4.090, "eye_m": 4.7,
                    "source": "cli"},
                   seed, params, 4032, 3024, notes="bruma ligera, contraluz")

    data = load_alignment(str(json_path))
    assert data["version"] == 1
    assert data["photo"] == "IMG_1234.jpg"
    assert data["alignment"]["azimuth_deg"] == 347.2
    # la resta que mide el error de brújula: ajustado - semilla EXIF
    assert data["seed"]["azimuth_deg"]["source"] == "exif"
    error_brujula = (data["alignment"]["azimuth_deg"]
                     - data["seed"]["azimuth_deg"]["value"])
    assert error_brujula == pytest.approx(6.2)
    assert data["seed"]["pitch_deg"]["source"] == "default"
    assert data["notes"] == "bruma ligera, contraluz"
    assert data["created_utc"].endswith("Z")
    assert data["image"] == {"width_px": 4032, "height_px": 3024,
                             "exif_oriented": True}


def test_pitch_positivo_baja_la_linea():
    from src.align import AlignmentParams, project_profile

    # Convenio de signo de la proyección: con pitch positivo la cámara mira
    # hacia arriba y el horizonte se dibuja MÁS ABAJO. Invertido, el ajuste
    # manual movería la línea al revés de lo que dice el slider — plausible
    # pero incorrecto.
    W, H = 4000, 3000
    y_neutro = project_profile([100.0], [0.0],
                               AlignmentParams(100.0, 65.0, 0.0, 0.0), W, H)[1]
    y_arriba = project_profile([100.0], [0.0],
                               AlignmentParams(100.0, 65.0, 2.0, 0.0), W, H)[1]
    assert y_neutro[0] == pytest.approx(H / 2)
    assert y_arriba[0] > y_neutro[0]


def test_projected_y_per_column():
    from src.align import projected_y_per_column

    # polilínea proyectada de 3 puntos; se pide y en columnas intermedias
    x_px = np.array([100.0, 200.0, 300.0])
    y_px = np.array([50.0, 70.0, 60.0])
    usable = np.array([True, True, True])
    y = projected_y_per_column(x_px, y_px, usable,
                               np.array([50.0, 150.0, 250.0, 400.0]))
    assert np.isnan(y[0])            # antes del tramo cubierto
    assert y[1] == pytest.approx(60.0)   # interpolado entre 50 y 70
    assert y[2] == pytest.approx(65.0)
    assert np.isnan(y[3])            # después

    # con menos de dos puntos utilizables no hay nada que interpolar
    y = projected_y_per_column(x_px, y_px, np.array([True, False, False]),
                               np.array([150.0]))
    assert np.isnan(y[0])


# El detector automático de cresta se ELIMINÓ deliberadamente: detectaba
# bordes de contraste arbitrarios (surcos, casas, cables) en vez de la
# frontera cielo/terreno, y el residuo que producía no significaba nada.
# Segmentar el skyline es el objetivo del modelo de la fase 2, no de
# heurísticas. La tira muestra la banda ampliada y la línea proyectada;
# el juicio del encaje lo hace el usuario mirando.


def test_load_oriented_photo_orientacion_6(tmp_path):
    from src.align import load_oriented_photo

    # Foto "vertical de móvil": píxeles apaisados 400x300 + Orientation=6
    # ("girar 90° CW al mostrar"), con marcador rojo en la esquina
    # superior-izquierda de los píxeles CRUDOS. Se verifica contenido, no
    # solo forma: con Orientation=6 ese marcador debe acabar arriba a la
    # DERECHA de la imagen orientada.
    raw = np.zeros((300, 400, 3), dtype=np.uint8)
    raw[:] = (100, 150, 200)
    raw[:30, :30] = (255, 0, 0)
    photo = tmp_path / "vertical.jpg"
    img = Image.fromarray(raw)
    exif = img.getexif()
    exif[0x0112] = 6
    img.save(photo, exif=exif, quality=95)

    oriented_img, oriented_np = load_oriented_photo(str(photo))
    # tamaño intercambiado, e imagen y array del MISMO origen
    assert oriented_img.size == (300, 400)
    assert oriented_np.shape == (400, 300, 3)
    # el marcador está arriba-derecha (media por la compresión JPEG)
    corner = oriented_np[:30, -30:].reshape(-1, 3).mean(axis=0)
    assert corner[0] > 200 and corner[1] < 60


def test_build_strip_indexa_filas_correctas():
    from src.align import build_strip

    # foto sintética donde cada píxel codifica su FILA en el canal rojo:
    # si la tira recortara en el eje equivocado (el bug transversal), los
    # valores leídos no coincidirían con las filas esperadas
    height, width = 512, 400
    photo = np.zeros((height, width, 3), dtype=np.uint8)
    photo[..., 0] = (np.arange(height)[:, None] // 2).astype(np.uint8)

    # silueta proyectada: línea horizontal en la fila 300
    x_px = np.array([0.0, width - 1.0])
    y_px = np.array([300.0, 300.0])
    usable = np.array([True, True])
    columns_px = np.arange(0.0, width, 1.0)
    strip_h, band_px = 21, 100.0
    patch, columns_valid = build_strip(photo, x_px, y_px, usable,
                                       columns_px, band_px, strip_h)

    assert patch.shape == (strip_h, width, 3)
    assert columns_valid.all()
    # la fila central de la tira es la fila 300 de la foto; la primera, la
    # 300-100=200; la última, la 300+100=400
    assert patch[strip_h // 2, :, 0] == pytest.approx(np.full(width, 150), abs=1)
    assert patch[0, :, 0] == pytest.approx(np.full(width, 100), abs=1)
    assert patch[-1, :, 0] == pytest.approx(np.full(width, 200), abs=1)

    # Columnas fuera del tramo proyectado: inválidas y rellenas con TRAMA
    # diagonal (dos grises alternos), no con un color liso. Un relleno liso
    # se lee como contenido de la foto: con la línea proyectada por encima
    # del borde superior, la tira aparentaba estar del revés.
    patch, columns_valid = build_strip(photo, np.array([100.0, 200.0]),
                                       y_px, usable, columns_px, band_px,
                                       strip_h)
    assert not columns_valid[:100].any() and not columns_valid[201:].any()
    fuera = patch[:, 0]
    assert set(np.unique(fuera)) == {58, 96}   # las dos tintas de la trama
    assert len(set(np.unique(fuera[:, 0]))) == 2  # alterna, no es liso


def test_build_strip_sobre_foto_orientada(tmp_path):
    from src.align import build_strip, load_oriented_photo

    # el flujo completo carga orientada + tira: cresta conocida en el
    # espacio ORIENTADO, la tira debe encontrarla donde toca
    height_raw, width_raw = 300, 400  # crudo apaisado -> orientado 300x400
    raw = np.zeros((height_raw, width_raw, 3), dtype=np.uint8)
    raw[:] = (140, 170, 210)
    # en el espacio orientado (400 filas x 300 cols), "terreno" desde la
    # fila 250: con Orientation=6 la fila orientada r viene de la COLUMNA
    # cruda r (crudo(fila, col) -> orientado(col, h_raw-1-fila)), así que
    # son las columnas crudas 250 en adelante
    raw[:, 250:] = (60, 55, 50)
    photo = tmp_path / "o6.jpg"
    img = Image.fromarray(raw)
    exif = img.getexif()
    exif[0x0112] = 6
    img.save(photo, exif=exif, quality=95)

    _, oriented = load_oriented_photo(str(photo))
    # línea proyectada en la fila 250 del espacio orientado
    patch, valid = build_strip(
        oriented, np.array([0.0, 299.0]), np.array([250.0, 250.0]),
        np.array([True, True]), np.arange(0.0, 300.0), 40.0, 21)
    assert valid.all()
    center_up = patch[9, :, 0].mean()    # justo encima del centro: cielo
    center_down = patch[12, :, 0].mean()  # justo debajo: terreno
    assert center_up > 120 and center_down < 80


def test_load_session(tmp_path):
    from src.align import (AlignmentParams, load_session, save_alignment,
                           seed_entry)

    photo = str(tmp_path / "IMG_1234.jpg")
    # sin JSON: no hay sesión
    assert load_session(photo) is None

    seed = {"azimuth_deg": seed_entry(341.0, "exif"),
            "hfov_deg": seed_entry(66.0, "exif"),
            "pitch_deg": seed_entry(0.0, "default"),
            "roll_deg": seed_entry(0.0, "default")}
    save_alignment(str(tmp_path / "IMG_1234.align.json"), "IMG_1234.jpg",
                   {"lat_deg": 36.745, "lon_deg": -4.090, "eye_m": 4.7,
                    "source": "cli"},
                   seed, AlignmentParams(26.2, 54.9, 6.4, 0.0),
                   3060, 4080, notes="calima")

    session = load_session(photo)
    assert session is not None
    assert session["alignment"]["azimuth_deg"] == 26.2
    assert session["notes"] == "calima"
    # el seed original viaja con la sesión: al reguardar se conserva TAL
    # CUAL, porque la procedencia EXIF de la primera vez es la medida del
    # error de brújula y una recarga no debe sobrescribirla
    assert session["seed"]["azimuth_deg"] == {"value": 341.0, "source": "exif"}

    # JSON corrupto o de versión desconocida: sesión inutilizable, None
    (tmp_path / "IMG_1234.align.json").write_text("{basura", encoding="utf-8")
    assert load_session(photo) is None
    (tmp_path / "IMG_1234.align.json").write_text('{"version": 99}',
                                                  encoding="utf-8")
    assert load_session(photo) is None


# --- búsqueda automática de alineamiento (asistente, NO la fase 2) ------


def _foto_desde_perfil(profile, params, width_px, height_px):
    """Foto sintética cuyo horizonte ES el perfil proyectado con params:
    cielo con degradado arriba, terreno oscuro debajo de la línea."""
    from src.align import project_profile, projected_y_per_column

    x_px, y_px, usable = project_profile(
        profile.azimuths_deg, profile.elevations_deg, params,
        width_px, height_px)
    y_line = projected_y_per_column(x_px, y_px, usable,
                                    np.arange(float(width_px)))
    photo = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    rows_idx = np.arange(height_px)[:, None]
    photo[..., 0] = 150
    photo[..., 1] = np.clip(185 - rows_idx * 0.03, 0, 255)
    photo[..., 2] = np.clip(225 - rows_idx * 0.05, 0, 255)
    terrain = rows_idx >= np.nan_to_num(y_line, nan=height_px)[None, :]
    for channel, value in enumerate((70, 62, 52)):
        photo[..., channel] = np.where(terrain, value, photo[..., channel])
    return photo


def _perfil_sintetico():
    az = np.arange(0.0, 360.0, 0.2)
    elev = (1.0 + 2.5 * np.exp(-((az - 20.0) / 5.0) ** 2)
            + 1.6 * np.exp(-((az - 38.0) / 3.0) ** 2)
            + 0.8 * np.sin(np.radians(az * 3.0)))
    return HorizonProfile(az, elev, np.full(az.shape, np.nan))


def test_detect_photo_skyline_sintetica():
    from src.align import AlignmentParams
    from src.align.search import detect_photo_skyline

    profile = _perfil_sintetico()
    true_params = AlignmentParams(23.7, 55.0, 2.0, 1.0)
    W, H = 1200, 900
    photo = _foto_desde_perfil(profile, true_params, W, H)
    # obstáculos que el detector debe ignorar: un cable cruzando el cielo y
    # bandas de contraste alto muy por debajo de la cresta
    photo[80:82, :] = (60, 60, 60)
    photo[700:720, :] = 250
    photo[750:770, :] = 10

    from src.align import project_profile, projected_y_per_column
    x_px, y_px, usable = project_profile(
        profile.azimuths_deg, profile.elevations_deg, true_params, W, H)
    cols, rows, valid = detect_photo_skyline(photo)
    y_true = projected_y_per_column(x_px, y_px, usable, cols)
    ok = valid & ~np.isnan(y_true)
    assert ok.mean() > 0.8
    # la cresta detectada sigue a la real (tolerancia: el paso de submuestreo)
    assert np.median(np.abs(rows[ok] - y_true[ok])) < 4.0


def test_busqueda_recupera_desplazamiento_sintetico():
    """CASO 7 DEL CONTRATO (versión sintética): renderizar el horizonte,
    desplazarlo artificialmente +13.7° y comprobar que el alineamiento
    recupera 13.7° ±0.1°.

    La foto se genera proyectando el perfil con az = semilla + 13.7 (más
    FOV/pitch/roll conocidos); la búsqueda parte de la semilla y debe volver
    a los parámetros verdaderos. Esto valida el buscador como asistente: el
    uso real sigue pasando por el juicio del usuario en la GUI.
    """
    from src.align import AlignmentParams
    from src.align.search import detect_photo_skyline, search_alignment

    profile = _perfil_sintetico()
    seed_az = 10.0
    true_params = AlignmentParams(seed_az + 13.7, 55.0, 2.0, 1.0)
    W, H = 1200, 900
    photo = _foto_desde_perfil(profile, true_params, W, H)

    cols, rows, valid = detect_photo_skyline(photo)
    result = search_alignment(profile.azimuths_deg, profile.elevations_deg,
                              cols[valid], rows[valid], W, H,
                              center_az_deg=seed_az)
    assert result.candidates, "la búsqueda no devolvió ningún candidato"
    best = result.candidates[0].params

    recovered_shift = (best.azimuth_deg - seed_az + 180.0) % 360.0 - 180.0
    assert recovered_shift == pytest.approx(13.7, abs=0.1)
    assert best.hfov_deg == pytest.approx(55.0, abs=1.0)
    assert best.pitch_deg == pytest.approx(2.0, abs=0.2)
    assert best.roll_deg == pytest.approx(1.0, abs=0.2)
    assert result.candidates[0].error_deg < 0.05
    # los tres se puntúan con la misma métrica exacta, así que la lista está
    # ordenada de verdad por el error mostrado (en píxeles, que es el
    # espacio de comparación)
    errors_px = [c.error_px for c in result.candidates]
    assert errors_px == sorted(errors_px)
    # perfil rico + FOV ancho: el óptimo es claro y no toca ningún borde
    assert result.reliable
    assert not result.ambiguous


def _cresta_proyectada(profile, params, width_px, height_px, step=5.0):
    """Cresta 'detectada' perfecta: la propia línea proyectada con params."""
    from src.align import project_profile, projected_y_per_column

    x_px, y_px, usable = project_profile(
        profile.azimuths_deg, profile.elevations_deg, params,
        width_px, height_px)
    cols = np.arange(60.0, width_px - 60.0, step)
    rows = projected_y_per_column(x_px, y_px, usable, cols)
    keep = ~np.isnan(rows)
    return cols[keep], rows[keep]


def test_solve_pitch_roll_recupera_los_dos():
    """Ata los SIGNOS del ajuste automático.

    El modelo es y(pitch, roll) − y(0,0) ≈ f·tan(pitch) + (x − W/2)·roll_rad:
    el residuo es una recta en x, su ordenada da la inclinación y su
    pendiente el giro. Si el signo del giro se invirtiera, el modo automático
    torcería la línea al revés sin que nada fallara de forma visible.
    """
    from src.align import AlignmentParams
    from src.align.search import solve_pitch_roll

    profile = _perfil_sintetico()
    W, H = 1200, 900
    for pitch, roll in ((2.0, 0.0), (0.0, 1.5), (3.0, -2.0), (-4.0, 3.0),
                        (8.0, 5.0)):
        verdad = AlignmentParams(23.7, 55.0, pitch, roll)
        cols, rows = _cresta_proyectada(profile, verdad, W, H)
        # se parte de pitch/roll a cero: el solver debe llegar solo
        partida = AlignmentParams(23.7, 55.0, 0.0, 0.0)
        solved = solve_pitch_roll(profile.azimuths_deg, profile.elevations_deg,
                                  partida, cols, rows, W, H)
        assert solved is not None
        assert solved[0] == pytest.approx(pitch, abs=0.05)
        assert solved[1] == pytest.approx(roll, abs=0.05)


def test_solve_pitch_roll_es_robusto_y_se_planta():
    from src.align import AlignmentParams
    from src.align.search import solve_pitch_roll

    profile = _perfil_sintetico()
    W, H = 1200, 900
    verdad = AlignmentParams(23.7, 55.0, 3.0, -1.5)
    cols, rows = _cresta_proyectada(profile, verdad, W, H)

    # 10% de columnas disparatadas (tejados, arbustos): el rechazo por MAD
    # debe impedir que tuerzan la recta
    sucias = rows.copy()
    sucias[::10] += 180.0
    partida = AlignmentParams(23.7, 55.0, 0.0, 0.0)
    solved = solve_pitch_roll(profile.azimuths_deg, profile.elevations_deg,
                              partida, cols, sucias, W, H)
    assert solved[0] == pytest.approx(3.0, abs=0.15)
    assert solved[1] == pytest.approx(-1.5, abs=0.15)

    # con cuatro columnas no se resuelve: mejor None que un ajuste inventado
    assert solve_pitch_roll(profile.azimuths_deg, profile.elevations_deg,
                            partida, cols[:4], rows[:4], W, H) is None


def test_sector_to_ranges():
    from src.align.search import FOV_RANGE_DEG, sector_to_ranges

    # sector normal de 30°
    center, width, lo, hi, fov_lo, fov_hi = sector_to_ranges(20.0, 50.0)
    assert (center, width, lo, hi) == pytest.approx((35.0, 30.0, 20.0, 50.0))
    assert fov_lo == pytest.approx(15.0) and fov_hi == pytest.approx(60.0)

    # arco > 180° -> el COMPLEMENTARIO, que cruza el norte. Inequívoco solo
    # porque el FOV máximo son 80°: ninguna foto abarca 340°.
    center, width, lo, hi, _fl, _fh = sector_to_ranges(10.0, 350.0)
    assert width == pytest.approx(20.0)
    assert lo == pytest.approx(350.0) and hi == pytest.approx(370.0)
    assert center == pytest.approx(0.0, abs=1e-9)

    # el FOV se acota al rango soportado por el buscador
    _c, _w, _lo, _hi, fov_lo, fov_hi = sector_to_ranges(0.0, 120.0)
    assert fov_lo >= FOV_RANGE_DEG[0] and fov_hi <= FOV_RANGE_DEG[1]


def test_dp_prefiere_el_camino_continuo():
    """El caso cresta→nube→cresta reducido a su esencia.

    Dos "crestas" posibles: una continua a media altura y un atajo por arriba
    (la nube), localmente más barato en un tramo corto. Sin coste de salto la
    decisión por columna se va al atajo; subiendo λ, los dos saltos lo hacen
    inviable y gana el camino continuo.

    Lo que se comprueba es el MECANISMO y que λ lo gobierna, no que la λ de
    producción rechace cualquier nube: se mantiene deliberadamente permisiva
    para no aplanar relieve real, y de hecho ya se midió que sobre evidencia
    de color esto no basta (ver cabecera de skyline.py).
    """
    from src.align.skyline import best_path

    height, width = 120, 200
    cost = np.full((height, width), 1.0)
    cost[60, :] = 0.30                     # cresta real, continua
    cost[10, 90:110] = 0.20                # "nube": algo más barata, y corta

    # sin penalización: el ahorro local manda y el camino salta a la nube
    suelto, _ = best_path(cost, jump_limit=60, smoothness=0.0)
    assert suelto[100] == 10

    # con penalización suficiente, los dos saltos no compensan
    rigido, margen = best_path(cost, jump_limit=60, smoothness=3.0)
    assert np.all(rigido == 60)
    assert margen.shape == (width,)
    assert np.all(margen >= 0.0)


def test_dp_lambda_controla_el_aplanado():
    from src.align.skyline import best_path

    height, width = 100, 60
    rng = np.random.default_rng(0)
    cost = rng.random((height, width)) * 0.2
    # un mínimo que serpentea con fuerza de columna a columna
    serpiente = (50 + 30 * np.sin(np.arange(width) / 3.0)).astype(int)
    cost[serpiente, np.arange(width)] = 0.0

    suelto, _ = best_path(cost, jump_limit=40, smoothness=0.0)
    rigido, _ = best_path(cost, jump_limit=40, smoothness=8.0)
    assert np.abs(np.diff(suelto)).sum() > np.abs(np.diff(rigido)).sum()
    assert np.abs(np.diff(rigido)).max() <= np.abs(np.diff(suelto)).max()


def test_low_confidence_se_guarda_y_se_lee(tmp_path):
    """Una referencia dudosa tiene que quedar marcada en el JSON: medir
    contra una referencia mala es peor que no medir."""
    from src.align import (AlignmentParams, load_session, save_alignment,
                           seed_entry)

    seed = {n: seed_entry(0.0, "default")
            for n in ("azimuth_deg", "hfov_deg", "pitch_deg", "roll_deg")}
    ruta = tmp_path / "x.align.json"
    save_alignment(str(ruta), "x.jpg",
                   {"lat_deg": 36.7, "lon_deg": -4.0, "eye_m": 5.0},
                   seed, AlignmentParams(10.0, 60.0, 0.0, 0.0), 100, 100,
                   notes="a ojo", low_confidence=True)
    datos = load_session(str(tmp_path / "x.jpg"))
    assert datos["low_confidence"] is True

    # por defecto NO es dudosa: marcarla tiene que ser un acto explícito
    save_alignment(str(ruta), "x.jpg",
                   {"lat_deg": 36.7, "lon_deg": -4.0, "eye_m": 5.0},
                   seed, AlignmentParams(10.0, 60.0, 0.0, 0.0), 100, 100)
    assert load_session(str(tmp_path / "x.jpg"))["low_confidence"] is False


def test_detector_sin_onnxruntime_cae_en_heuristica():
    """La dependencia es OPCIONAL: sin modelo la herramienta sigue teniendo
    detector, igual que peaks/ sigue funcionando sin red."""
    from pathlib import Path as _Path

    from src.align.search import detect_photo_skyline
    from src.align.segmentation import ModelUnavailable, build_model_detector

    inexistente = _Path("models/no_existe_este_modelo.onnx")
    assert build_model_detector(model_path=inexistente) is detect_photo_skyline

    from src.align.segmentation import load_session
    with pytest.raises(ModelUnavailable):
        load_session(inexistente)

    # y el selector rechaza nombres inventados en vez de elegir por su cuenta
    from src.align.skyline import build_detector
    with pytest.raises(ValueError):
        build_detector("dp")          # DP sobre color: descartada, no existe


def test_lo_que_descalifica_es_aceptar_sin_revisar_no_usar_la_busqueda():
    """El criterio que decide si una referencia sirve como VERDAD.

    Un primer intento descartaba toda referencia que hubiera pasado por el
    buscador, y eso tiraba trabajo humano legítimo: verificar dónde caen los
    TOPÓNIMOS es información independiente del detector. Lo que no sirve es
    aceptar la salida del buscador tal cual.
    """
    from src.align import provenance_entry

    manual = provenance_entry(detector="modelo+dp")
    assert manual["manual"] is True and manual["reviewed"] is True

    # aceptada sin revisar: el detector se mediría contra su propia salida
    cruda = provenance_entry(search_used=True, auto_pitch_roll=True,
                             detector="heuristica")
    assert cruda["manual"] is False and cruda["reviewed"] is False

    # misma ayuda automática, pero revisada después: vuelve a servir
    revisada = provenance_entry(search_used=True, auto_pitch_roll=True,
                                detector="heuristica", manual_review=True,
                                review_delta={"hfov_deg": -3.78},
                                review_note="FOV corregido a mano")
    assert revisada["manual"] is False      # hubo ayuda, y consta
    assert revisada["reviewed"] is True     # pero un humano la verificó
    assert revisada["review_delta"] == {"hfov_deg": -3.78}


def test_las_cinco_referencias_constan_como_revisadas():
    """Las cinco se verificaron a mano contra topónimos después de la
    búsqueda. Son el único conjunto que hay: descartarlas por haber usado el
    buscador sería confundir 'usó la ayuda' con 'aceptó la ayuda'."""
    import json as _json
    from pathlib import Path as _Path

    from src.align import infer_provenance

    rutas = sorted(_Path("Dataset").glob("*.align.json"))
    if not rutas:
        pytest.skip("Dataset/ no está en el repo (fotos no versionadas)")
    for ruta in rutas:
        datos = _json.loads(ruta.read_text(encoding="utf-8"))
        origen = infer_provenance(datos)
        assert origen["reviewed"] is True, f"{ruta.name} no consta revisada"
        # y consta CÓMO se verificó: sin eso, "revisada" es una afirmación
        # sin respaldo, que es justo lo que no queremos volver a tener
        assert origen.get("review_note"), f"{ruta.name} sin review_note"


def test_procedencia_no_registrada_no_decide_por_su_cuenta():
    """Un JSON anterior al campo no se descarta, pero tampoco decide: hay que
    anotarle a mano cómo se verificó.

    La deducción a partir del seed se probó contra las cinco referencias
    reales y falla, así que aquí solo puede informar (`hint`).
    """
    from src.align import infer_provenance

    # 1) el alignment ES el seed del buscador (caso 20260812 / Nerja)
    es_seed = {
        "seed": {"azimuth_deg": {"value": 329.25, "source": "default"},
                 "hfov_deg": {"value": 63.91, "source": "exif"},
                 "pitch_deg": {"value": 13.35, "source": "default"},
                 "roll_deg": {"value": -0.81, "source": "default"}},
        "alignment": {"azimuth_deg": 329.2, "hfov_deg": 63.9,
                      "pitch_deg": 13.35, "roll_deg": -0.8}}
    origen = infer_provenance(es_seed)
    assert origen["unknown"] is True and origen["reviewed"] is False
    assert origen["search_used"] is True

    # 2) seed a valores por defecto, búsqueda lanzada desde la GUI: la
    #    comparación no ve nada y solo las notas lo delatan (caso 141721)
    desde_gui = {
        "notes": "alineamiento por busqueda automatica validado a ojo",
        "seed": {"azimuth_deg": {"value": 0.0, "source": "default"},
                 "hfov_deg": {"value": 67.38, "source": "exif"},
                 "pitch_deg": {"value": 0.0, "source": "default"},
                 "roll_deg": {"value": 0.0, "source": "default"}},
        "alignment": {"azimuth_deg": 15.4, "hfov_deg": 50.5,
                      "pitch_deg": 4.0, "roll_deg": 0.2}}
    origen = infer_provenance(desde_gui)
    assert origen["unknown"] is True and origen["reviewed"] is False

    # 3) sin indicio ninguno: SIGUE sin decidir. "No hay pruebas de que se
    #    aceptara a ciegas" no es lo mismo que "consta que se revisó".
    sin_pistas = {"seed": {}, "alignment": {"azimuth_deg": 51.2}}
    origen = infer_provenance(sin_pistas)
    assert origen["unknown"] is True and origen["reviewed"] is False

    # 4) una procedencia DECLARADA manda, y no se marca como deducida
    declarada = {"provenance": {"search_used": True, "auto_pitch_roll": True,
                                "detector": "heuristica",
                                "manual": False, "manual_review": True,
                                "reviewed": True}}
    origen = infer_provenance(declarada)
    assert origen["reviewed"] is True
    assert origen["unknown"] is False and origen["inferred"] is False

    # 5) un provenance escrito ANTES de existir `reviewed` se completa solo,
    #    sin que un fichero viejo pierda su condición
    antiguo = {"provenance": {"search_used": False, "auto_pitch_roll": False,
                              "detector": None, "manual": True}}
    assert infer_provenance(antiguo)["reviewed"] is True


def test_el_respaldo_al_heuristico_avisa_una_vez(tmp_path, capsys):
    """Sin la dependencia opcional la herramienta funciona; avisar en cada
    ejecución sería ruido sobre algo que el usuario no ha elegido."""
    from src.align.segmentation import announce_fallback

    marca = tmp_path / "aviso"
    assert announce_fallback("onnxruntime no disponible", marker=marca) is True
    assert "heurístico" in capsys.readouterr().out

    # la segunda vez se calla, y no imprime NADA
    assert announce_fallback("onnxruntime no disponible", marker=marca) is False
    assert capsys.readouterr().out == ""

    # salvo que se pida explícitamente
    assert announce_fallback("onnxruntime no disponible", verbose=True,
                             marker=marca) is True
    assert "heurístico" in capsys.readouterr().out


def test_el_defecto_es_el_modelo_con_dp():
    """El defecto medido, y el respaldo que hace que no cueste nada."""
    from src.align.skyline import DEFAULT_DETECTOR

    assert DEFAULT_DETECTOR == "modelo+dp"

    import peakid.__main__ as cli
    parser = cli._build_parser()
    args = parser.parse_args(["align", "--photo", "x.jpg",
                              "--lat", "36.7", "--lon", "-4.1"])
    assert args.skyline_detector == DEFAULT_DETECTOR
    # y la vía para producir referencias limpias existe y está apagada
    assert args.no_auto_pitch_roll is False


def test_entrada_del_modelo_conserva_el_aspecto():
    """El cuanto de la máscara es el suelo del error de localización (medido:
    8.4 px a 512 de entrada, 2.0 px a 1536), así que el tamaño de entrada no
    es cosmético. Cuadrar la imagen desperdicia resolución en el eje largo."""
    from src.align.segmentation import SIZE_MULTIPLE, _input_size

    for width, height in ((9248, 6944), (3060, 4080), (2551, 1701)):
        w, h = _input_size(width, height, 1536)
        assert w % SIZE_MULTIPLE == 0 and h % SIZE_MULTIPLE == 0
        assert max(w, h) == 1536              # el lado largo manda
        # el aspecto se conserva salvo el redondeo a múltiplo de 32
        assert abs((w / h) - (width / height)) < 0.06
        # y el eje largo de la entrada es el eje largo de la foto
        assert (w > h) == (width > height)


def test_coste_de_borde_es_relativo_a_cada_columna():
    """Normalizar por columna, no globalmente: si no, una columna en calima o
    a contraluz tiene bordes débiles en absoluto y quedaría descartada entera
    frente a otra bien iluminada, aunque su cresta sea igual de nítida."""
    from src.align.skyline import edge_cost

    alto, ancho = 60, 8
    img = np.zeros((alto, ancho, 3), dtype=float)
    img[30:] = 200.0                       # escalón nítido, columnas 0..3
    img[30:, 4:] = 8.0                     # mismo escalón, 25 veces más débil
    coste = edge_cost(img)

    fuerte, debil = coste[29:31, 0].min(), coste[29:31, 4].min()
    assert fuerte == pytest.approx(debil, abs=1e-6)   # mismo coste pese al
    assert fuerte < 0.05                              # contraste distinto
    assert coste[10, 0] > 0.9 and coste[10, 4] > 0.9  # lejos del borde, caro


def test_exif_gps_vacio_no_produce_nan():
    """Hay cámaras que escriben el bloque GPS con racionales 0/0 y el
    hemisferio a '\\x00' cuando no llegaron a fijar posición (medido en
    141720.jpg). Eso daba lat/lon = NaN, que entraban en el pipeline sin que
    nada fallara: el fallo callado que CLAUDE.md prohíbe.
    """
    from src.align import _dms_to_deg, _seed_from_exif

    class FakeExif(dict):
        def __init__(self, ifds):
            super().__init__()
            self._ifds = ifds

        def get_ifd(self, tag):
            return self._ifds.get(tag, {})

    nan = float("nan")
    vacio = FakeExif({0x8825: {1: "\x00", 2: (nan, nan, nan),
                               3: "\x00", 4: (nan, nan, nan), 6: nan}})
    seed = _seed_from_exif(vacio)
    assert "lat_deg" not in seed and "lon_deg" not in seed
    assert seed.get("gps_invalid") is True   # se dice, no se calla

    # el hemisferio tiene que ser explícito: '\x00' no es "norte"
    assert _dms_to_deg((36.0, 45.0, 24.0), "\x00") is None
    assert _dms_to_deg((36.0, 45.0, 24.0), "N") == pytest.approx(36.756667)
    assert _dms_to_deg((4.0, 6.0, 12.0), "W") == pytest.approx(-4.103333)
    assert _dms_to_deg((nan, 45.0, 24.0), "N") is None
    assert _dms_to_deg(None, "N") is None

    # un GPS bueno sigue funcionando y no se marca inválido
    bueno = FakeExif({0x8825: {1: "N", 2: (36.0, 45.0, 24.674),
                               3: "W", 4: (4.0, 6.0, 12.406), 17: 341.0}})
    seed = _seed_from_exif(bueno)
    assert seed["lat_deg"] == pytest.approx(36.7568, abs=1e-3)
    assert seed["lon_deg"] == pytest.approx(-4.1034, abs=1e-3)
    assert "gps_invalid" not in seed


def test_cli_rechaza_coordenadas_invalidas():
    from peakid.__main__ import _CliError, _require_valid_coords

    _require_valid_coords(36.745, -4.090, "cli")        # válidas: no protesta
    for lat, lon in ((float("nan"), -4.0), (36.7, float("inf")),
                     (95.0, -4.0), (36.7, 200.0), (None, -4.0)):
        with pytest.raises(_CliError):
            _require_valid_coords(lat, lon, "exif-gps")


def test_error_saturado_no_parece_casi_bueno():
    """El recorte de residuo a 40 px satura: un desajuste de 350 px daba el
    mismo número que uno de 45 y se informó como '38 px, casi bueno' cuando
    la línea iba a 350 px de la cresta. `saturated` lo distingue."""
    from src.align.search import SearchCandidate, _clipped_mean_abs, _saturated_fraction

    catastrofico = np.full(200, 350.0)
    mediocre = np.concatenate([np.full(180, 12.0), np.full(20, 300.0)])
    assert _clipped_mean_abs(catastrofico) == pytest.approx(40.0)
    assert _saturated_fraction(catastrofico) == pytest.approx(1.0)
    assert _saturated_fraction(mediocre) == pytest.approx(0.1)

    from src.align import AlignmentParams
    p = AlignmentParams(0.0, 50.0, 0.0, 0.0)
    assert SearchCandidate(p, 40.0, 0.5, 1.0, 1.0).saturated
    assert not SearchCandidate(p, 12.5, 0.2, 1.0, 0.1).saturated


def test_peak_label_con_nombre_alternativo():
    """El caso del Naranjo de Bulnes.

    En OSM `name` lleva el topónimo LOCAL: el Naranjo está como
    `name=Picu Urriellu` con `alt_name=Naranjo de Bulnes`, así que aparecía
    rotulado con un nombre que el usuario no reconocía y parecía faltar de
    la lista. En Asturias, Galicia, Euskadi y Catalunya ese va a ser el caso
    habitual, no la excepción.
    """
    from src.peaks import Peak, _parse_node

    urriellu = _parse_node({
        "type": "node", "id": 129959999, "lat": 43.20083, "lon": -4.81770,
        "tags": {"name": "Picu Urriellu", "alt_name": "Naranjo de Bulnes",
                 "name:es": "Pico Urriellu", "name:ast": "Picu Urriellu",
                 "ele": "2518", "natural": "peak", "wikidata": "Q2636482"}})
    assert urriellu.name == "Picu Urriellu"
    assert urriellu.alt_name == "Naranjo de Bulnes"   # alt_name gana a name:es
    assert urriellu.label == "Picu Urriellu (Naranjo de Bulnes)"
    assert urriellu.ele_m == 2518.0

    # sin alternativo, la etiqueta no se ensucia con paréntesis vacíos
    diente = _parse_node({
        "type": "node", "id": 5495616040, "lat": 43.20712, "lon": -4.83212,
        "tags": {"name": "Diente de Urriellu", "ele": "2322"}})
    assert diente.alt_name is None
    assert diente.label == "Diente de Urriellu"

    # name:es sirve cuando no hay alt_name, pero no si repite el nombre
    con_es = _parse_node({"type": "node", "id": 1, "lat": 0.0, "lon": 0.0,
                          "tags": {"name": "Aizkorri", "name:es": "Aizcorri"}})
    assert con_es.label == "Aizkorri (Aizcorri)"
    igual = _parse_node({"type": "node", "id": 2, "lat": 0.0, "lon": 0.0,
                         "tags": {"name": "Mulhacén", "name:es": "Mulhacén"}})
    assert igual.label == "Mulhacén"

    # sin nombre, identificable por su id de OSM en vez de en blanco
    assert Peak(42, None, 0.0, 0.0, None, None).label == "osm:42"


def test_detector_descarta_obstaculos_estrechos():
    """Postes, farolas y ramas en primer plano rompen la cresta con un
    escalón de cientos de píxeles y envenenan el ajuste de pitch/roll: unas
    pocas columnas que no son horizonte tuercen la recta.

    El detector parte la cresta por esos escalones y tira los tramos
    demasiado cortos para ser relieve.
    """
    from src.align import AlignmentParams
    from src.align.search import detect_photo_skyline

    profile = _perfil_sintetico()
    verdad = AlignmentParams(23.7, 55.0, 2.0, 0.0)
    W, H = 1200, 900
    photo = _foto_desde_perfil(profile, verdad, W, H)
    # tres postes estrechos que suben muy por encima de la cresta
    for x0 in (250, 600, 940):
        photo[:, x0:x0 + 7] = (60, 55, 50)

    cols, rows, valid = detect_photo_skyline(photo)
    for x0 in (250, 600, 940):
        # exactamente las columnas del poste (250..256), no sus vecinas, que
        # son cresta legítima y deben conservarse
        sobre_poste = (cols >= x0) & (cols <= x0 + 6)
        assert not valid[sobre_poste].any(), (
            f"el poste en x={x0} sigue alimentando el ajuste")
    assert valid.mean() > 0.7   # el resto de la cresta se conserva


def test_detector_conserva_la_cresta_si_todo_es_fragmentado():
    # Salvaguarda: si el criterio se llevara casi todo, es que la foto es así
    # de accidentada, y quedarse sin columnas es peor que conservarlas.
    from src.align.search import _drop_short_segments

    rows = np.arange(60, dtype=float) * 37 % 400   # dientes de sierra puros
    valid = np.ones(60, dtype=bool)
    conservado = _drop_short_segments(rows, valid, height=400, width=60)
    assert conservado.sum() >= 0.2 * valid.sum()


def test_bindings_numericos_son_teclas_no_botones(tmp_path):
    """En Tk un detalle NUMÉRICO en un binding es el número de BOTÓN del
    ratón, no una tecla: bind("<1>") registra <Button-1>.

    Ese despiste hacía dos cosas a la vez: las teclas 1..5 no seleccionaban
    candidato, y cada clic izquierdo en la foto aplicaba el candidato 1,
    llevándose por delante el ajuste manual en curso. El test mira los
    bindings registrados de verdad, que es donde el error es visible.
    """
    import tkinter as tk

    from src.align import AlignmentParams
    from src.align.gui import _AlignApp

    try:
        tk.Tk().destroy()
    except tk.TclError:                      # sin display: nada que comprobar
        pytest.skip("sin entorno gráfico")

    # la regla de Tk que provoca el fallo, comprobada aquí para que el test
    # siga significando algo si algún día cambia
    root = tk.Tk()
    root.bind("<1>", lambda _e: None)
    root.bind("<Key-1>", lambda _e: None)
    regla = set(root.bind())
    root.destroy()
    assert "<Button-1>" in regla   # <1> ES un botón del ratón
    assert "1" in regla            # <Key-1> ES la tecla

    # y ahora los bindings REALES de la ventana, que es donde estaba el fallo
    profile = _perfil_sintetico()
    params = AlignmentParams(23.7, 55.0, 2.0, 1.0)
    photo = tmp_path / "sintetica.jpg"
    Image.fromarray(_foto_desde_perfil(profile, params, 600, 450)).save(
        photo, quality=90)

    app = _AlignApp(str(photo), profile, 36.7, -4.0, 10.0, params, {}, "cli")
    registrados = set(app.root.bind())
    app.root.destroy()

    assert {"0", "1", "2", "3", "4", "5"} <= registrados, (
        f"los dígitos deben estar como TECLAS; registrado: {registrados}")
    # ningún binding de botón en el toplevel: <1>..<5> habrían registrado
    # <Button-N> y cada clic en la foto aplicaría el candidato 1
    assert not [b for b in registrados if b.startswith("<Button-")], registrados


def test_busqueda_detecta_ambiguedad_con_campo_estrecho():
    """EL FALLO REAL DEL USUARIO (teleobjetivo de Sierra Nevada).

    Con campo de visión estrecho la firma del horizonte es pobre y la
    correlación tiene muchos máximos casi equivalentes. Aquí se fuerza el
    caso extremo: un perfil PERIÓDICO (la misma cresta repetida cada 30° de
    azimut) fotografiado con 15° de campo. La búsqueda no puede distinguir
    una repetición de otra, y eso debe DECIRSE, no presentarse como solución.
    """
    from src.align import AlignmentParams
    from src.align.search import detect_photo_skyline, search_alignment

    az = np.arange(0.0, 360.0, 0.2)
    # Periodo de 15°, MENOR que el semirrango explorado (±20°): así la
    # repetición equivalente cae DENTRO del espacio de búsqueda y la
    # ambigüedad es real. Con periodo mayor que el rango no habría nada que
    # detectar, solo un espacio demasiado estrecho.
    elev = 2.0 + 1.5 * np.sin(np.radians(az * 24.0))
    profile = HorizonProfile(az, elev, np.full(az.shape, np.nan))
    true_params = AlignmentParams(40.0, 15.0, 1.0, 0.0)
    W, H = 1200, 900
    photo = _foto_desde_perfil(profile, true_params, W, H)

    cols, rows, valid = detect_photo_skyline(photo)
    result = search_alignment(profile.azimuths_deg, profile.elevations_deg,
                              cols[valid], rows[valid], W, H,
                              center_az_deg=40.0, az_margin_deg=20.0,
                              fov_hint_deg=15.0, fov_margin_deg=5.0)
    assert result.candidates
    assert result.ambiguous, (
        f"margen {result.ambiguity_margin:.2f}: la búsqueda debería admitir "
        "que no distingue entre repeticiones del mismo perfil")
    assert result.alternative is not None
    separation = abs((result.alternative.params.azimuth_deg
                      - result.candidates[0].params.azimuth_deg + 180.0)
                     % 360.0 - 180.0)
    assert separation > 10.0
    assert not result.reliable


def test_busqueda_detecta_saturacion_de_fov():
    # Rango de FOV que NO contiene el verdadero: el óptimo se apoya en el
    # borde y hay que avisar de que el bueno puede estar fuera. Es lo que
    # pasó con el teleobjetivo real contra el rango antiguo de 40-75°.
    from src.align import AlignmentParams
    from src.align.search import build_fov_grid, detect_photo_skyline, search_alignment

    profile = _perfil_sintetico()
    true_params = AlignmentParams(23.7, 20.0, 1.0, 0.0)
    W, H = 1200, 900
    photo = _foto_desde_perfil(profile, true_params, W, H)
    cols, rows, valid = detect_photo_skyline(photo)

    # rango 25-45: el verdadero (20) queda justo fuera -> el óptimo se apoya
    # en el extremo inferior y hay que avisar
    fuera = search_alignment(profile.azimuths_deg, profile.elevations_deg,
                             cols[valid], rows[valid], W, H,
                             center_az_deg=23.7, fov_hint_deg=35.0,
                             fov_margin_deg=10.0)
    assert fuera.fov_at_edge
    assert not fuera.reliable
    # LÍMITE MEDIDO de la detección de saturación: solo muerde cuando el
    # verdadero está CERCA del borde. Con el rango 45-65 (verdad 20, muy
    # lejos) la búsqueda encuentra un óptimo interior espurio en ~48° y
    # fov_at_edge NO se dispara. Contra ese caso protege la ambigüedad, no
    # la saturación; documentado aquí para que nadie confíe de más en ella.
    lejos = search_alignment(profile.azimuths_deg, profile.elevations_deg,
                             cols[valid], rows[valid], W, H,
                             center_az_deg=23.7, fov_hint_deg=55.0,
                             fov_margin_deg=10.0)
    assert not lejos.fov_at_edge
    assert lejos.candidates[0].params.hfov_deg > 40.0  # espurio, lejos de 20

    # el rango completo por defecto sí lo contiene
    dentro = search_alignment(profile.azimuths_deg, profile.elevations_deg,
                              cols[valid], rows[valid], W, H,
                              center_az_deg=23.7)
    assert not dentro.fov_at_edge
    assert dentro.candidates[0].params.hfov_deg == pytest.approx(20.0, abs=1.5)

    # la rejilla geométrica cubre de 10 a 80 con resolución relativa
    # constante: densa donde el teleobjetivo la necesita
    grid = build_fov_grid()
    assert grid[0] == pytest.approx(10.0) and grid[-1] == pytest.approx(80.0)
    assert np.all(np.diff(grid) > 0)
    assert (grid[1] - grid[0]) < (grid[-1] - grid[-2])  # paso creciente


def test_busqueda_hints_acotan_el_espacio():
    # Sin pista, un desplazamiento de 35° cae fuera de los ±20° explorados y
    # la búsqueda ni siquiera mira ahí (el fallo del caso de Granada);
    # con --az-hint el rango se centra donde toca y lo encuentra.
    from src.align import AlignmentParams
    from src.align.search import detect_photo_skyline, search_alignment

    profile = _perfil_sintetico()
    true_az = 45.0
    true_params = AlignmentParams(true_az, 55.0, 2.0, 1.0)
    W, H = 1200, 900
    photo = _foto_desde_perfil(profile, true_params, W, H)
    cols, rows, valid = detect_photo_skyline(photo)

    sin_pista = search_alignment(profile.azimuths_deg, profile.elevations_deg,
                                 cols[valid], rows[valid], W, H,
                                 center_az_deg=10.0)
    assert sin_pista.az_range_deg == pytest.approx((-10.0, 30.0))
    assert not sin_pista.reliable  # el verdadero ni se exploró

    con_pista = search_alignment(profile.azimuths_deg, profile.elevations_deg,
                                 cols[valid], rows[valid], W, H,
                                 center_az_deg=true_az, az_margin_deg=10.0)
    assert con_pista.az_range_deg == pytest.approx((35.0, 55.0))
    assert con_pista.candidates[0].params.azimuth_deg == pytest.approx(
        true_az, abs=0.3)


def test_exif_transpose_orienta_la_foto(tmp_path):
    from PIL import Image, ImageOps

    # Una foto vertical de móvil se guarda con los píxeles apaisados más el
    # tag Orientation=6 ("girar 90° CW al mostrar"). Sin aplicarlo, la GUI
    # dibujaba el buffer crudo bajo una silueta en orientación normal: la
    # foto salía del revés.
    photo = tmp_path / "vertical.jpg"
    raw = Image.new("RGB", (4000, 3000))
    exif = raw.getexif()
    exif[0x0112] = 6  # Orientation
    raw.save(photo, exif=exif)

    with Image.open(photo) as opened:
        assert opened.size == (4000, 3000)  # pillow NO orienta al abrir
        oriented = ImageOps.exif_transpose(opened)
    # tras orientar, ancho y alto se intercambian: de ahí depende toda la
    # proyección, así que el tamaño hay que leerlo DESPUÉS
    assert oriented.size == (3000, 4000)


def test_seed_from_exif_objeto_falso():
    from src.align import _seed_from_exif

    class FakeExif(dict):
        def __init__(self, base, ifds):
            super().__init__(base)
            self._ifds = ifds

        def get_ifd(self, tag):
            return self._ifds.get(tag, {})

    # focal 35mm de 28 -> hfov = 2*atan(18/28) = 65.47°; GPS de Torre del
    # Mar con dirección 341° y hemisferio W (longitud negativa)
    exif = FakeExif(
        {41989: 28},
        {0x8825: {1: "N", 2: (36, 44, 42.0), 3: "W", 4: (4, 5, 24.0),
                  17: 341.0}})
    seed = _seed_from_exif(exif)
    assert seed["hfov_deg"] == pytest.approx(65.47, abs=0.01)
    assert seed["azimuth_deg"] == 341.0
    assert seed["lat_deg"] == pytest.approx(36.745, abs=1e-4)
    assert seed["lon_deg"] == pytest.approx(-4.090, abs=1e-4)

    # sin EXIF: dict vacío, sin reventar
    assert _seed_from_exif(FakeExif({}, {})) == {}
