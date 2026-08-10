"""Picos de OpenStreetMap: consulta Overpass, recolocación sobre el DEM y
evaluación de visibilidad desde un observador.

Roles de cada fuente (ver CLAUDE.md): OSM da la posición y la altitud OFICIAL
de cada cima; el DEM solo pone el terreno intermedio del rayo y sirve de
respaldo de altitud cuando el tag `ele` falta. La visibilidad se comprueba
contra la altitud de OSM, nunca contra la del DEM (SRTM subestima cimas).
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from src.dem import TileNotFoundError, elevation_m, elevations_m
from src.geo import (
    R_EARTH_M,
    azimuth_deg,
    destination_points_deg,
    elevation_deg,
    elevations_deg,
    haversine_m,
)
from src.horizon import (
    STEP_M,
    SUMMIT_MARGIN_M,
    Visibility,
    VisibilityResult,
    _truncation_index,
)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RELOCATE_RADIUS_M = 200.0
RELOCATE_STEP_M = 15.0  # medio píxel SRTM1: no se escapa ningún nodo de rejilla

_M_PER_DEG = math.radians(1.0) * R_EARTH_M  # metros por grado de latitud


@dataclass
class Peak:
    """Un nodo natural=peak de OSM, parseado."""
    osm_id: int
    name: str | None
    lat_deg: float
    lon_deg: float
    ele_m: float | None
    wikidata: str | None


@dataclass
class PeakSighting:
    """Un pico evaluado desde el observador. Solo VISIBLE o UNKNOWN: los
    BLOCKED se excluyen (esa sí es una respuesta definitiva)."""
    peak: Peak
    lat_deg: float  # recolocada (o la original de OSM si no mejoró)
    lon_deg: float
    h_m: float
    h_source: str  # "osm" (tag ele) | "dem" (respaldo)
    azimuth_deg: float
    distance_m: float
    elevation_deg: float
    visibility: Visibility
    truncated_at_m: float | None


def fetch_peaks(lat_deg: float, lon_deg: float, radius_m: float,
                cache_dir: str = "cache") -> list[Peak]:
    """Nodos natural=peak alrededor de (lat_deg, lon_deg), con caché en disco.

    La respuesta de Overpass se guarda CRUDA en
    cache/overpass/peaks_{lat}_{lon}_r{radius}.json — el nombre es la clave,
    determinista sobre los parámetros redondeados (4 decimales, ~11 m), así
    dos llamadas desde casi el mismo sitio comparten caché. Sin TTL: las
    montañas no se mueven; refrescar = borrar el fichero.
    """
    lat_q, lon_q = round(lat_deg, 4), round(lon_deg, 4)
    radius = int(radius_m)
    cache_path = (Path(cache_dir) / "overpass"
                  / f"peaks_{lat_q:.4f}_{lon_q:.4f}_r{radius}.json")
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        query = ('[out:json][timeout:60];'
                 f'node["natural"="peak"](around:{radius},{lat_q:.4f},{lon_q:.4f});'
                 'out body;')
        payload = _overpass_request(query)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return [_parse_node(element) for element in payload.get("elements", [])
            if element.get("type") == "node"]


def relocate_peak(peak: Peak, dem_dir: str = "data") -> tuple[float, float]:
    """Punto más alto del DEM a menos de 200 m de la coordenada OSM.

    Las coordenadas de OSM están puestas a ojo y pueden desviarse decenas de
    metros; recolocar corrige la POSICIÓN para que el rayo de visibilidad
    apunte a la cima y no a la ladera. No corrige la altitud: esa sigue las
    reglas de _peak_height.

    Solo se mueve si el DEM del mejor punto supera al de la coordenada
    original (empate = quedarse). Si el DEM no puede responder (voids, tile
    ausente), se devuelve la coordenada original: la recolocación es una
    corrección, no un requisito.
    """
    steps = int(RELOCATE_RADIUS_M / RELOCATE_STEP_M)
    offsets_m = np.arange(-steps, steps + 1) * RELOCATE_STEP_M  # incluye el 0
    east_m, north_m = np.meshgrid(offsets_m, offsets_m)
    in_circle = east_m**2 + north_m**2 <= RELOCATE_RADIUS_M**2
    east_m, north_m = east_m[in_circle], north_m[in_circle]

    lat_arr = peak.lat_deg + north_m / _M_PER_DEG
    lon_arr = peak.lon_deg + east_m / (_M_PER_DEG * math.cos(math.radians(peak.lat_deg)))
    try:
        heights_m = elevations_m(lat_arr, lon_arr, dem_dir)
        here_m = elevation_m(peak.lat_deg, peak.lon_deg, dem_dir)
    except TileNotFoundError:
        return peak.lat_deg, peak.lon_deg
    if np.all(np.isnan(heights_m)):
        return peak.lat_deg, peak.lon_deg

    best = int(np.nanargmax(heights_m))
    if here_m is not None and heights_m[best] <= here_m:
        return peak.lat_deg, peak.lon_deg
    return float(lat_arr[best]), float(lon_arr[best])


def evaluate_peaks(peaks: list[Peak], lat_obs_deg: float, lon_obs_deg: float,
                   h_obs_m: float, *, max_distance_m: float | None = None,
                   min_ele_m: float | None = None, named_only: bool = True,
                   dem_dir: str = "data") -> list[PeakSighting]:
    """Recoloca, filtra y evalúa visibilidad. Sin red: recibe los picos ya
    descargados (o fabricados, en tests).

    h_obs_m es la altitud del OJO sobre el nivel del mar, no la cota del
    suelo (mismo convenio que horizon/).

    Devuelve solo VISIBLE y UNKNOWN, ordenados por azimut (el orden natural
    para cruzar con la foto y el perfil), distancia como desempate. Los picos
    sin altitud evaluable (sin tag ele Y sin DEM utilizable) se excluyen.
    """
    exists_cache: dict[tuple[int, int], bool] = {}
    sightings = []
    for peak in peaks:
        if named_only and not peak.name:
            continue
        lat_p, lon_p = relocate_peak(peak, dem_dir)
        height = _peak_height(peak, lat_p, lon_p, dem_dir)
        if height is None:
            continue
        h_m, h_source = height
        if min_ele_m is not None and h_m < min_ele_m:
            continue
        d_m = haversine_m(lat_obs_deg, lon_obs_deg, lat_p, lon_p)
        if max_distance_m is not None and d_m > max_distance_m:
            continue
        bearing_deg = azimuth_deg(lat_obs_deg, lon_obs_deg, lat_p, lon_p)
        target_elev_deg = elevation_deg(d_m, h_obs_m, h_m)
        result = _ray_visibility(
            lat_obs_deg, lon_obs_deg, h_obs_m, bearing_deg, d_m,
            target_elev_deg, dem_dir, exists_cache)
        if result.status is Visibility.BLOCKED:
            continue
        sightings.append(PeakSighting(
            peak=peak, lat_deg=lat_p, lon_deg=lon_p, h_m=h_m,
            h_source=h_source, azimuth_deg=bearing_deg, distance_m=d_m,
            elevation_deg=target_elev_deg, visibility=result.status,
            truncated_at_m=result.truncated_at_m))
    sightings.sort(key=lambda s: (s.azimuth_deg, s.distance_m))
    return sightings


def visible_peaks(lat_obs_deg: float, lon_obs_deg: float, h_obs_m: float, *,
                  radius_m: float = 100_000.0,
                  max_distance_m: float | None = None,
                  min_ele_m: float | None = None, named_only: bool = True,
                  dem_dir: str = "data",
                  cache_dir: str = "cache") -> list[PeakSighting]:
    """Picos visibles (o UNKNOWN) desde el observador: Overpass + caché +
    recolocación + visibilidad. h_obs_m = altitud del ojo."""
    peaks = fetch_peaks(lat_obs_deg, lon_obs_deg, radius_m, cache_dir)
    if max_distance_m is None:
        max_distance_m = radius_m
    return evaluate_peaks(peaks, lat_obs_deg, lon_obs_deg, h_obs_m,
                          max_distance_m=max_distance_m, min_ele_m=min_ele_m,
                          named_only=named_only, dem_dir=dem_dir)


def _peak_height(peak: Peak, lat_deg: float, lon_deg: float,
                 dem_dir: str) -> tuple[float, str] | None:
    """Altitud a usar para la visibilidad: el tag ele de OSM si existe (es la
    oficial), el DEM en la coordenada recolocada como respaldo. NUNCA
    max(ele, DEM): mezclar fuentes difuminaría la separación de roles, y un
    DEM por encima del ele oficial suele ser la celda promediando una loma
    vecina, no una cima más alta."""
    if peak.ele_m is not None:
        return peak.ele_m, "osm"
    try:
        dem_h = elevation_m(lat_deg, lon_deg, dem_dir)
    except TileNotFoundError:
        return None
    if dem_h is None:
        return None
    return dem_h, "dem"


def _ray_visibility(lat_obs_deg: float, lon_obs_deg: float, h_obs_m: float,
                    bearing_deg: float, d_total_m: float,
                    target_elev_deg: float, dem_dir: str,
                    exists_cache: dict[tuple[int, int], bool],
                    ) -> VisibilityResult:
    """check_visibility vectorizado por rayo, para evaluar cientos de picos
    (~2-3 ms/rayo frente a ~0.2 s del escalar). Misma semántica: BLOCKED si
    alguna muestra antes del objetivo iguala o supera su elevación, UNKNOWN
    si el rayo sale de cobertura sin bloqueo previo, VISIBLE si no. Los voids
    (NaN) se saltan. Un test comprueba que coincide con el escalar.
    """
    distances_m = np.arange(STEP_M, d_total_m - SUMMIT_MARGIN_M, STEP_M)
    if distances_m.size == 0:
        return VisibilityResult(Visibility.VISIBLE)
    lat_arr, lon_arr = destination_points_deg(
        lat_obs_deg, lon_obs_deg, bearing_deg, distances_m)

    cut = _truncation_index(lat_arr, lon_arr, dem_dir, exists_cache)
    if cut is not None:
        lat_arr, lon_arr = lat_arr[:cut], lon_arr[:cut]
        ray_distances_m = distances_m[:cut]
    else:
        ray_distances_m = distances_m

    if ray_distances_m.size:
        terrain_m = elevations_m(lat_arr, lon_arr, dem_dir)
        sample_elev_deg = elevations_deg(ray_distances_m, h_obs_m, terrain_m)
        with np.errstate(invalid="ignore"):  # NaN >= x es False: void se salta
            blocking = np.flatnonzero(sample_elev_deg >= target_elev_deg)
        if blocking.size:
            return VisibilityResult(
                Visibility.BLOCKED,
                blocked_at_m=float(ray_distances_m[blocking[0]]))
    if cut is not None:
        return VisibilityResult(Visibility.UNKNOWN,
                                truncated_at_m=float(distances_m[cut]))
    return VisibilityResult(Visibility.VISIBLE)


def _overpass_request(query: str) -> dict:
    """POST a Overpass con un único reintento ante rate limit (429/504).

    La política de uso de Overpass exige un User-Agent identificable; con el
    de python-requests por defecto responde 406.
    """
    headers = {"User-Agent": "peakid/0.1 (identificacion geometrica de cimas)"}
    for attempt in (0, 1):
        response = requests.post(OVERPASS_URL, data={"data": query},
                                 headers=headers, timeout=60)
        if response.status_code in (429, 504) and attempt == 0:
            time.sleep(30)
            continue
        response.raise_for_status()
        return response.json()
    raise AssertionError("unreachable")


def _parse_node(element: dict) -> Peak:
    tags = element.get("tags", {})
    return Peak(
        osm_id=int(element["id"]),
        name=tags.get("name"),
        lat_deg=float(element["lat"]),
        lon_deg=float(element["lon"]),
        ele_m=_parse_ele(tags.get("ele")),
        wikidata=tags.get("wikidata"),
    )


def _parse_ele(raw: str | None) -> float | None:
    """El tag ele lleva de todo ("2069", "2069 m", basura). Lo que no parse
    se trata como ausente, nunca se inventa."""
    if raw is None:
        return None
    text = raw.strip().lower().removesuffix("m").strip()
    try:
        return float(text)
    except ValueError:
        return None
