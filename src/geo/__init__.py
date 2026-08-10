"""Geometría básica: distancia, azimut, curvatura+refracción y elevación.

Convenios (ver CLAUDE.md): coordenadas (lat, lon) en grados decimales,
azimut horario desde el norte geográfico en [0, 360), distancias y
alturas en metros. Radianes solo dentro del cuerpo de las funciones.
"""

import math

R_EARTH_M = 6_371_000.0  # radio terrestre medio
K_REFRACTION = 0.13      # coeficiente de refracción atmosférica estándar


def haversine_m(lat_a_deg: float, lon_a_deg: float,
                lat_b_deg: float, lon_b_deg: float) -> float:
    """Distancia en metros entre A y B sobre la esfera de radio R_EARTH_M."""
    lat_a_rad = math.radians(lat_a_deg)
    lat_b_rad = math.radians(lat_b_deg)
    dlat_rad = math.radians(lat_b_deg - lat_a_deg)
    dlon_rad = math.radians(lon_b_deg - lon_a_deg)
    a = (math.sin(dlat_rad / 2.0) ** 2
         + math.cos(lat_a_rad) * math.cos(lat_b_rad) * math.sin(dlon_rad / 2.0) ** 2)
    return 2.0 * R_EARTH_M * math.asin(math.sqrt(a))


def azimuth_deg(lat_a_deg: float, lon_a_deg: float,
                lat_b_deg: float, lon_b_deg: float) -> float:
    """Azimut inicial de A hacia B: horario desde el norte, en [0, 360)."""
    lat_a_rad = math.radians(lat_a_deg)
    lat_b_rad = math.radians(lat_b_deg)
    dlon_rad = math.radians(lon_b_deg - lon_a_deg)
    # atan2(este, norte): ángulo desde el norte abriéndose hacia el este
    # = horario. atan2(norte, este) daría el convenio matemático (girado).
    east = math.sin(dlon_rad) * math.cos(lat_b_rad)
    north = (math.cos(lat_a_rad) * math.sin(lat_b_rad)
             - math.sin(lat_a_rad) * math.cos(lat_b_rad) * math.cos(dlon_rad))
    return math.degrees(math.atan2(east, north)) % 360.0


def curvature_drop_m(d_m: float) -> float:
    """Caída aparente en metros por curvatura terrestre, corregida por
    refracción atmosférica (siempre incluida, no es opcional)."""
    return (1.0 - K_REFRACTION) * d_m ** 2 / (2.0 * R_EARTH_M)


def elevation_deg(d_m: float, h_a_m: float, h_b_m: float) -> float:
    """Ángulo de elevación del objetivo B (altura h_b_m) a distancia d_m,
    visto desde el observador A (altura h_a_m).

    Incluye siempre curvatura+refracción. Negativo si B queda bajo el
    plano horizontal del observador.
    """
    return math.degrees(math.atan2(h_b_m - h_a_m - curvature_drop_m(d_m), d_m))
