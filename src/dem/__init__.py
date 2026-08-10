"""Lectura e interpolación de tiles SRTM (.hgt).

Convenios (ver CLAUDE.md): fila 0 = borde norte del tile, columna 0 = borde
oeste. El nombre del fichero indica la esquina SUROESTE (p. ej. N40W004.hgt
cubre lat 40..41, lon -4..-3). Valor -32768 = void, nunca se usa como altitud.
"""

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VOID = -32768

_SRTM1_BYTES = 25_934_402  # 3601 x 3601 x 2
_SRTM1_N = 3600
_SRTM3_BYTES = 2_884_802  # 1201 x 1201 x 2
_SRTM3_N = 1200

_TILE_NAME_RE = re.compile(r"^([NS])(\d{2})([EW])(\d{3})\.hgt$")


class TileNotFoundError(FileNotFoundError):
    """El .hgt esperado para unas coordenadas no está en dem_dir."""


@dataclass
class Tile:
    array: np.memmap  # shape (n+1, n+1), dtype '>i2' (int16 big-endian)
    lat_sw: int
    lon_sw: int
    n: int  # tamaño - 1 (3600 en SRTM1, 1200 en SRTM3)


_tile_cache: dict[str, Tile] = {}


def _parse_tile_name(filename: str) -> tuple[int, int]:
    match = _TILE_NAME_RE.match(filename)
    if not match:
        raise ValueError(f"nombre de tile no reconocido: {filename!r}")
    lat_hem, lat_str, lon_hem, lon_str = match.groups()
    lat_sw = int(lat_str) * (1 if lat_hem == "N" else -1)
    lon_sw = int(lon_str) * (1 if lon_hem == "E" else -1)
    return lat_sw, lon_sw


def load_tile(path: str) -> Tile:
    """Carga un tile .hgt, deduciendo la resolución (SRTM1/SRTM3) del
    tamaño exacto del fichero."""
    size_bytes = os.path.getsize(path)
    if size_bytes == _SRTM1_BYTES:
        n = _SRTM1_N
    elif size_bytes == _SRTM3_BYTES:
        n = _SRTM3_N
    else:
        raise ValueError(
            f"{path}: tamaño {size_bytes} bytes no coincide con SRTM1 "
            f"({_SRTM1_BYTES}) ni SRTM3 ({_SRTM3_BYTES})"
        )
    side = n + 1
    array = np.memmap(path, dtype=">i2", mode="r", shape=(side, side))
    lat_sw, lon_sw = _parse_tile_name(os.path.basename(path))
    return Tile(array=array, lat_sw=lat_sw, lon_sw=lon_sw, n=n)


def locate_tile_path(lat_deg: float, lon_deg: float) -> str:
    """Nombre del fichero .hgt (sin ruta) que debería cubrir (lat_deg, lon_deg).

    Usa floor, no int(): para longitudes negativas truncar hacia cero da la
    esquina SO equivocada (int(-4.10) = -4 -> "W004", pero el tile correcto
    es "W005" porque floor(-4.10) = -5 es la esquina SO real).
    """
    lat_sw = math.floor(lat_deg)
    lon_sw = math.floor(lon_deg)
    lat_hem = "N" if lat_sw >= 0 else "S"
    lon_hem = "E" if lon_sw >= 0 else "W"
    return f"{lat_hem}{abs(lat_sw):02d}{lon_hem}{abs(lon_sw):03d}.hgt"


def get_tile(lat_deg: float, lon_deg: float, dem_dir: str = "data") -> Tile:
    """Tile cacheado que cubre (lat_deg, lon_deg), cargándolo si hace falta."""
    filename = locate_tile_path(lat_deg, lon_deg)
    if filename in _tile_cache:
        return _tile_cache[filename]
    path = Path(dem_dir) / filename
    if not path.exists():
        raise TileNotFoundError(
            f"no se encontró {filename} en {dem_dir!r} "
            f"(requerido para lat={lat_deg}, lon={lon_deg})"
        )
    tile = load_tile(str(path))
    _tile_cache[filename] = tile
    return tile


def elevation_m(lat_deg: float, lon_deg: float, dem_dir: str = "data") -> float | None:
    """Altitud interpolada bilinealmente en (lat_deg, lon_deg).

    None si las cuatro esquinas usadas en la interpolación son void
    (-32768); las esquinas void individuales se excluyen y se renormalizan
    los pesos en vez de mezclar el número en la media.
    """
    tile = get_tile(lat_deg, lon_deg, dem_dir)

    row = tile.n * (tile.lat_sw + 1 - lat_deg)
    col = tile.n * (lon_deg - tile.lon_sw)
    row0 = min(max(math.floor(row), 0), tile.n)
    col0 = min(max(math.floor(col), 0), tile.n)
    frow = row - row0
    fcol = col - col0
    row1 = min(row0 + 1, tile.n)
    col1 = min(col0 + 1, tile.n)

    corners = (
        (tile.array[row0, col0], (1 - frow) * (1 - fcol)),
        (tile.array[row0, col1], (1 - frow) * fcol),
        (tile.array[row1, col0], frow * (1 - fcol)),
        (tile.array[row1, col1], frow * fcol),
    )
    valid = [(float(v), w) for v, w in corners if v != VOID]
    if not valid:
        return None

    weight_sum = sum(w for _, w in valid)
    if weight_sum > 0:
        return sum(v * w for v, w in valid) / weight_sum
    # degenerado: el punto cae casi exacto sobre un nodo void y las demás
    # esquinas válidas tienen peso ~0 -> media simple sin peso
    return sum(v for v, _ in valid) / len(valid)
