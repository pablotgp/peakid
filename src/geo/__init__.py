"""Geometría básica: distancia, azimut, curvatura+refracción y elevación.

Convenios (ver CLAUDE.md): coordenadas (lat, lon) en grados decimales,
azimut horario desde el norte geográfico en [0, 360), distancias y
alturas en metros. Radianes solo dentro del cuerpo de las funciones.
"""

import math

import numpy as np

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


def destination_point_deg(lat_deg: float, lon_deg: float,
                          bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """Coordenada alcanzada partiendo de (lat_deg, lon_deg) con rumbo
    bearing_deg (horario desde el norte) tras recorrer distance_m sobre el
    círculo máximo.

    Es el problema geodésico directo, complementario de haversine_m/azimuth_deg.
    """
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    bearing_rad = math.radians(bearing_deg)
    ang_rad = distance_m / R_EARTH_M
    lat2_rad = math.asin(
        math.sin(lat_rad) * math.cos(ang_rad)
        + math.cos(lat_rad) * math.sin(ang_rad) * math.cos(bearing_rad)
    )
    lon2_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(ang_rad) * math.cos(lat_rad),
        math.cos(ang_rad) - math.sin(lat_rad) * math.sin(lat2_rad),
    )
    return math.degrees(lat2_rad), (math.degrees(lon2_rad) + 180.0) % 360.0 - 180.0


def destination_points_deg(lat_deg: float, lon_deg: float, bearing_deg: float,
                           distances_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """destination_point_deg para un array de distancias con el mismo rumbo.

    El barrido de horizonte llama a esto una vez por rayo en vez de una vez
    por muestra. Un test comprueba que coincide con la versión escalar.
    """
    ang_rad = np.asarray(distances_m, dtype=float) / R_EARTH_M
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    bearing_rad = math.radians(bearing_deg)
    lat2_rad = np.arcsin(
        math.sin(lat_rad) * np.cos(ang_rad)
        + math.cos(lat_rad) * np.sin(ang_rad) * math.cos(bearing_rad)
    )
    lon2_rad = lon_rad + np.arctan2(
        math.sin(bearing_rad) * np.sin(ang_rad) * math.cos(lat_rad),
        np.cos(ang_rad) - math.sin(lat_rad) * np.sin(lat2_rad),
    )
    return np.degrees(lat2_rad), (np.degrees(lon2_rad) + 180.0) % 360.0 - 180.0


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


def elevations_deg(d_m: np.ndarray, h_a_m: float, h_b_m: np.ndarray) -> np.ndarray:
    """elevation_deg sobre arrays (una llamada por rayo en vez de por muestra).

    Propaga NaN: una muestra de terreno NaN (void) da elevación NaN, que quien
    llame debe descartar con np.nanmax, no con np.max.
    """
    d = np.asarray(d_m, dtype=float)
    return np.degrees(np.arctan2(np.asarray(h_b_m, dtype=float) - h_a_m
                                 - curvature_drop_m(d), d))
