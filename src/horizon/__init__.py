"""Rayos, visibilidad y barrido de horizonte de 360°.

Roles de cada fuente (ver CLAUDE.md): la altitud del objetivo llega como
parámetro (viene de OSM, que es la fuente oficial de cimas); el DEM se
consulta SOLO para el terreno intermedio del rayo, que a 30 m sí está bien
representado. SRTM subestima las cimas por promediado.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from src.dem import TileNotFoundError, elevation_m, elevations_m, locate_tile_path
from src.geo import (
    azimuth_deg,
    destination_point_deg,
    destination_points_deg,
    elevation_deg,
    elevations_deg,
    haversine_m,
)

STEP_M = 30.0               # paso de muestreo (≈ resolución del SRTM1)
MAX_DISTANCE_M = 150_000.0  # más allá casi nunca hay visibilidad real
AZIMUTH_STEP_DEG = 0.2      # 1800 rayos en el barrido de 360°
# Los últimos metros del rayo se excluyen de la comprobación de visibilidad:
# a esa distancia del objetivo el "terreno" ES la ladera del propio objetivo
# (mismo radio de incertidumbre que usa la recolocación de picos), y contarlo
# como obstáculo hace que una cima se bloquee a sí misma por centésimas de
# grado. Un obstáculo real a <200 m de una cima lejana subtiende ~0.001°.
SUMMIT_MARGIN_M = 200.0


class Visibility(Enum):
    VISIBLE = "visible"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"  # el rayo salió de la cobertura DEM sin bloquearse antes


@dataclass
class VisibilityResult:
    status: Visibility
    truncated_at_m: float | None = None  # solo si status == UNKNOWN
    blocked_at_m: float | None = None    # solo si status == BLOCKED: distancia
                                         # del primer obstáculo del rayo


@dataclass
class HorizonProfile:
    azimuths_deg: np.ndarray     # (1800,) 0.0, 0.2, ..., 359.8
    elevations_deg: np.ndarray   # (1800,) máximo del rayo; NaN si todo era void
    truncated_at_m: np.ndarray   # (1800,) NaN si el rayo llegó a max_km


def check_visibility(lat_obs_deg: float, lon_obs_deg: float, h_obs_m: float,
                     lat_tgt_deg: float, lon_tgt_deg: float, h_tgt_m: float,
                     dem_dir: str = "data") -> VisibilityResult:
    """¿Se ve el objetivo desde el observador?

    h_obs_m es la altitud del OJO sobre el nivel del mar, no la cota del
    suelo: quien llame debe sumar la altura de la persona (~1.7 m). Pasar la
    cota del terreno hace que las muestras cercanas bloqueen espuriamente.

    h_tgt_m es la altitud oficial del objetivo (OSM), no la del DEM.

    Devuelve tres estados en vez de un bool: UNKNOWN significa que el rayo se
    quedó sin cobertura DEM antes de encontrar obstáculo, e incluye la
    distancia de truncamiento. Un hueco de cobertura no puede disfrazarse de
    VISIBLE.
    """
    d_total_m = haversine_m(lat_obs_deg, lon_obs_deg, lat_tgt_deg, lon_tgt_deg)
    if d_total_m <= STEP_M:
        return VisibilityResult(Visibility.VISIBLE)

    bearing_deg = azimuth_deg(lat_obs_deg, lon_obs_deg, lat_tgt_deg, lon_tgt_deg)
    target_elevation_deg = elevation_deg(d_total_m, h_obs_m, h_tgt_m)

    # Muestras en orden creciente de distancia. Si el terreno bloquea antes
    # del hueco de cobertura, BLOCKED corta el bucle y el hueco ni se
    # consulta: el orden resuelve solo la precedencia. El rayo se detiene
    # SUMMIT_MARGIN_M antes del objetivo (ver el comentario de la constante).
    d_m = STEP_M
    while d_m < d_total_m - SUMMIT_MARGIN_M:
        lat_s_deg, lon_s_deg = destination_point_deg(
            lat_obs_deg, lon_obs_deg, bearing_deg, d_m)
        try:
            terrain_m = elevation_m(lat_s_deg, lon_s_deg, dem_dir)
        except TileNotFoundError:
            return VisibilityResult(Visibility.UNKNOWN, truncated_at_m=d_m)
        # void puntual dentro de un tile que sí existe: se salta, no cuenta
        # ni como obstáculo ni como terreno libre
        if terrain_m is not None:
            if elevation_deg(d_m, h_obs_m, terrain_m) >= target_elevation_deg:
                return VisibilityResult(Visibility.BLOCKED, blocked_at_m=d_m)
        d_m += STEP_M

    return VisibilityResult(Visibility.VISIBLE)


def horizon_profile(lat_deg: float, lon_deg: float, h_obs_m: float,
                    max_km: float = 150.0,
                    dem_dir: str = "data") -> HorizonProfile:
    """Barrido de 360° en pasos de 0.2°: ángulo de elevación máximo del
    terreno en cada azimut.

    h_obs_m es la altitud del OJO sobre el nivel del mar, no la cota del
    suelo (ver check_visibility).

    Sin corte anticipado: un pico lejano puede superar a uno cercano más
    bajo, así que hay que recorrer el rayo entero. Los rayos que salen de la
    cobertura DEM se truncan y lo registran en truncated_at_m.
    """
    distances_m = np.arange(STEP_M, max_km * 1000.0 + STEP_M / 2.0, STEP_M)
    azimuths_deg = np.arange(0.0, 360.0, AZIMUTH_STEP_DEG)
    elevations = np.full(azimuths_deg.shape, np.nan)
    truncated = np.full(azimuths_deg.shape, np.nan)
    exists_cache: dict[tuple[int, int], bool] = {}

    for i, bearing_deg in enumerate(azimuths_deg):
        lat_s_deg, lon_s_deg = destination_points_deg(
            lat_deg, lon_deg, float(bearing_deg), distances_m)

        cut = _truncation_index(lat_s_deg, lon_s_deg, dem_dir, exists_cache)
        if cut is not None:
            truncated[i] = distances_m[cut]
            lat_s_deg, lon_s_deg = lat_s_deg[:cut], lon_s_deg[:cut]
            ray_distances_m = distances_m[:cut]
        else:
            ray_distances_m = distances_m
        if ray_distances_m.size == 0:
            continue

        terrain_m = elevations_m(lat_s_deg, lon_s_deg, dem_dir)
        ray_elevations_deg = elevations_deg(ray_distances_m, h_obs_m, terrain_m)
        # nanmax, no max: los voids llegan como NaN y envenenarían el máximo
        # del azimut entero. Si TODO el rayo es void, el perfil es NaN, no un
        # número inventado.
        if not np.all(np.isnan(ray_elevations_deg)):
            elevations[i] = np.nanmax(ray_elevations_deg)

    return HorizonProfile(azimuths_deg=azimuths_deg,
                          elevations_deg=elevations,
                          truncated_at_m=truncated)


def _truncation_index(lat_arr: np.ndarray, lon_arr: np.ndarray, dem_dir: str,
                      exists_cache: dict[tuple[int, int], bool]) -> int | None:
    """Índice de la primera muestra EN ORDEN DE DISTANCIA cuyo tile no está
    descargado, o None si el rayo entero tiene cobertura.

    Por orden de distancia, no por grupo de tile: un rayo puede cruzar el
    tile A, luego el B, y rozar A otra vez en una esquina. Se descarta todo
    lo posterior al corte aunque caiga en tiles que sí están.
    """
    lat_sw_arr = np.floor(lat_arr).astype(np.int64)
    lon_sw_arr = np.floor(lon_arr).astype(np.int64)
    # agrupación por tile en un único int para poder usar np.unique 1-D; la
    # codificación es local a esta función (no necesita coincidir con la de
    # dem/) y no colisiona porque lon_sw ∈ [-180, 180], de rango < 512
    keys = lat_sw_arr * 512 + lon_sw_arr
    unique_keys, representative = np.unique(keys, return_index=True)

    missing = np.zeros(lat_arr.shape, dtype=bool)
    for key, rep in zip(unique_keys, representative):
        corner = (int(lat_sw_arr[rep]), int(lon_sw_arr[rep]))
        if corner not in exists_cache:
            name = locate_tile_path(corner[0] + 0.5, corner[1] + 0.5)
            exists_cache[corner] = (Path(dem_dir) / name).exists()
        if not exists_cache[corner]:
            missing |= keys == key
    first = np.flatnonzero(missing)
    return int(first[0]) if first.size else None
