"""Tests dorados — contrato del proyecto.

NO modificar los valores esperados sin aprobación explícita del usuario
en el chat. Si un test falla, el bug está en el código (ver CLAUDE.md).
"""

import math
from pathlib import Path

import pytest

from src.dem import TileNotFoundError, elevation_m, load_tile, locate_tile_path
from src.geo import azimuth_deg, curvature_drop_m, elevation_deg, haversine_m

# (lat_deg, lon_deg, altitud_m)
SOL = (40.4168, -3.7038, 650.0)
PENALARA = (40.8508, -3.9578, 2428.0)

DEM_DIR = "data"
PENALARA_TILE = Path(DEM_DIR) / "N40W004.hgt"  # ausente: proyecto movido a Málaga


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
