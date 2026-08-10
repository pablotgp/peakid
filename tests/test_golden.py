"""Tests dorados — contrato del proyecto.

NO modificar los valores esperados sin aprobación explícita del usuario
en el chat. Si un test falla, el bug está en el código (ver CLAUDE.md).
"""

import math

import pytest

from src.geo import azimuth_deg, curvature_drop_m, elevation_deg, haversine_m

# (lat_deg, lon_deg, altitud_m)
SOL = (40.4168, -3.7038, 650.0)
PENALARA = (40.8508, -3.9578, 2428.0)


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
