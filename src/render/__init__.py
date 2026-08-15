"""Render del perfil de horizonte a PNG.

Solo pillow + numpy (sin matplotlib, ver CLAUDE.md). No hay test dorado de
este módulo: la verificación es visual.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.horizon import HorizonProfile, Visibility
from src.peaks import PeakSighting

# Paleta validada con el validador CVD del skill dataviz: el par
# terreno/truncado separa dE 23.6 (protan) y 28.5 (visión normal), ambos con
# contraste >= 3:1 sobre el fondo. La silueta es deliberadamente casi neutra
# (no es un color de serie categórica). "Sin dato" lleva trama diagonal como
# codificación secundaria además del color.
_BG = (252, 252, 251)
_SURFACE = (255, 255, 255)
_TERRAIN = (61, 74, 99)          # silueta con cobertura completa
_TERRAIN_EDGE = (44, 55, 75)
_TRUNCATED = (192, 118, 27)      # rayo cortado por falta de tile: no fiable
_TRUNCATED_EDGE = (143, 87, 20)
_NODATA_BAND = (237, 238, 241)   # todo el rayo era void: sin dato
_NODATA_HATCH = (170, 177, 189)
_GRID = (228, 230, 234)
_HORIZON = (107, 114, 128)
_INK = (55, 65, 81)
_INK_MUTED = (107, 114, 128)

_CARDINALS = ((0.0, "N"), (90.0, "E"), (180.0, "S"), (270.0, "O"), (360.0, "N"))

# clases de sector
_NODATA, _TRUNC, _OK = 0, 1, 2

_LABEL_BAND_1X = 150  # alto (a 1x) de la banda de etiquetas cuando hay picos


@dataclass
class LabelReport:
    """Qué picos consiguieron etiqueta y cuáles no, en orden de prioridad.

    `discarded` y `out_of_range` se mantienen separados a propósito: perder la
    competencia por una ranura y quedar fuera del encuadre son cosas
    distintas. Ninguna desaparece en silencio."""
    placed: list[str]
    discarded: list[str]
    out_of_range: list[str]


def render_horizon_png(profile: HorizonProfile, path: str,
                       peaks: list[PeakSighting] | None = None,
                       width_px: int = 1600,
                       height_px: int = 520,
                       az_min_deg: float | None = None,
                       az_max_deg: float | None = None) -> LabelReport | None:
    """Dibuja el perfil de horizonte como silueta rellena y lo guarda en path.

    Tres clases de sector: terreno fiable (silueta oscura), rayo truncado por
    falta de cobertura DEM (ámbar: el máximo es un mínimo garantizado, puede
    haber terreno más alto detrás del corte) y sin dato (banda gris con trama:
    el rayo entero era void).

    Si se pasan picos (la lista de peaks.evaluate_peaks/visible_peaks), cada
    uno etiquetado dibuja una guía vertical desde su punto en la silueta y su
    nombre+altitud rotado 90° en la banda superior. Los que colisionan con
    etiquetas de mayor prioridad se descartan; devuelve el informe de qué
    quedó y qué se tiró (None si no se pasaron picos).

    az_min_deg/az_max_deg recortan el dibujo a un sector, reescalando el eje X
    al rango y filtrando las etiquetas. Se dan ambos o ninguno. El sector
    puede cruzar el norte (340→40); sin recorte el resultado es idéntico al
    de antes de existir esta opción.
    """
    if (az_min_deg is None) != (az_max_deg is None):
        raise ValueError("az_min_deg y az_max_deg se pasan juntos o ninguno")
    cropped = az_min_deg is not None
    az_min, az_max_u = _crop_span(az_min_deg or 0.0,
                                  az_max_deg if cropped else 360.0)
    span_deg = az_max_u - az_min

    s = 2  # supersampling: se dibuja a 2x y se reduce con LANCZOS
    width, height = width_px * s, height_px * s
    margin_l, margin_r = 64 * s, 20 * s
    margin_t = 56 * s + (_LABEL_BAND_1X * s if peaks else 0)
    margin_b = 44 * s
    x0, x1 = margin_l, width - margin_r
    y0, y1 = margin_t, height - margin_b
    plot_w, plot_h = x1 - x0, y1 - y0

    az_full = np.asarray(profile.azimuths_deg, dtype=float)
    # el paso sale del array ORIGINAL: un sector estrecho puede dejar una sola
    # muestra, y ahí az[1]-az[0] reventaría
    az_step_deg = az_full[1] - az_full[0] if az_full.size > 1 else 0.2
    az, elev, trunc = _crop_profile(
        az_full, np.asarray(profile.elevations_deg, dtype=float),
        np.asarray(profile.truncated_at_m, dtype=float),
        az_min, az_max_u, az_step_deg)

    classes = np.where(np.isnan(elev), _NODATA,
                       np.where(~np.isnan(trunc), _TRUNC, _OK))

    # escala Y: autoescala a los datos, siempre incluyendo el 0 (la línea de
    # horizonte es la referencia del gráfico) más un margen del 8 %
    finite = elev[~np.isnan(elev)]
    if finite.size:
        lo, hi = min(float(finite.min()), 0.0), max(float(finite.max()), 0.0)
    else:
        lo, hi = -1.0, 1.0
    pad = max((hi - lo) * 0.08, 0.05)
    lo, hi = lo - pad, hi + pad

    def x_at(az_deg: float) -> float:
        """az_deg en espacio DESENROLLADO (ver _unwrap_deg). Clampado al
        marco porque pillow no recorta: sin esto la silueta del borde se
        saldría del área de dibujo."""
        x = x0 + (az_deg - az_min) / span_deg * plot_w
        return min(max(x, x0), x1)

    def y_at(elev_deg: float) -> float:
        return y0 + (hi - elev_deg) / (hi - lo) * plot_h

    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], fill=_SURFACE)
    font = _font(12 * s)
    font_cardinal = _font(13 * s)

    # bandas "sin dato" (debajo de todo, a plot completo) con trama diagonal
    for start, end in _runs(classes, _NODATA):
        bx0, bx1 = x_at(az[start]), x_at(az[end - 1] + az_step_deg)
        draw.rectangle([bx0, y0, bx1, y1], fill=_NODATA_BAND)
        _hatch(img, int(bx0), y0, int(bx1), y1, spacing=9 * s,
               color=_NODATA_HATCH, line_w=s)

    # rejilla: verticales tenues cada 45°, horizontales tenues en los ticks Y
    ticks = _nice_ticks(lo, hi)
    for tick in ticks:
        if tick != 0.0:
            ty = y_at(tick)
            draw.line([x0, ty, x1, ty], fill=_GRID, width=s)
    az_ticks = _azimuth_ticks(az_min, az_max_u, cropped)
    for tick_az, _ in az_ticks:
        gx = x_at(tick_az)
        draw.line([gx, y0, gx, y1], fill=_GRID, width=s)

    # silueta: un polígono por tramo contiguo de la misma clase
    for cls, fill, edge in ((_OK, _TERRAIN, _TERRAIN_EDGE),
                            (_TRUNC, _TRUNCATED, _TRUNCATED_EDGE)):
        for start, end in _runs(classes, cls):
            top = [(x_at(az[i]), y_at(elev[i])) for i in range(start, end)]
            top.append((x_at(az[end - 1] + az_step_deg), y_at(elev[end - 1])))
            polygon = top + [(top[-1][0], y1), (top[0][0], y1)]
            draw.polygon(polygon, fill=fill)
            draw.line(top, fill=edge, width=2 * s)

    # línea de horizonte (0°): encima de la silueta, discontinua para que se
    # distinga de la rejilla y no se pierda sobre el relleno oscuro
    hy = y_at(0.0)
    _dashed_line(draw, x0, hy, x1, hy, dash=6 * s, gap=4 * s,
                 fill=_HORIZON, width=s)

    # marco y ejes
    draw.rectangle([x0, y0, x1, y1], outline=_GRID, width=s)

    for tick_az, label in az_ticks:
        if label is None:
            continue
        tx = x_at(tick_az)
        draw.line([tx, y1, tx, y1 + 5 * s], fill=_INK_MUTED, width=s)
        draw.text((tx, y1 + 8 * s), label, font=font_cardinal,
                  fill=_INK, anchor="ma")

    for tick in ticks:
        ty = y_at(tick)
        draw.line([x0 - 5 * s, ty, x0, ty], fill=_INK_MUTED, width=s)
        draw.text((x0 - 9 * s, ty), _fmt_deg(tick), font=font,
                  fill=_INK if tick == 0.0 else _INK_MUTED, anchor="rm")

    _legend(draw, x0, 22 * s, font, scale=s,
            with_unknown_peaks=bool(peaks) and any(
                p.visibility is Visibility.UNKNOWN for p in peaks))

    report = None
    if peaks:
        report = _place_labels(img, draw, peaks, x_at, y_at,
                               band_top=56 * s, band_bottom=y0 - 2 * s,
                               y_plot=(y0, y1), font=font, s=s,
                               az_min=az_min, az_max_u=az_max_u)

    img = img.resize((width_px, height_px), Image.LANCZOS)
    img.save(path, "PNG")
    return report


def _crop_span(az_min_deg: float, az_max_deg: float) -> tuple[float, float]:
    """Normaliza a [0,360) y DESENROLLA el extremo superior.

    Si az_max <= az_min el sector cruza el norte y se le suman 360: 340→40
    devuelve (340, 400), un intervalo de 60° ya monótono. az_min == az_max se
    interpreta como vuelta completa desde az_min (span 360), no como sector
    vacío, que sería inútil.
    """
    az_min = az_min_deg % 360.0
    az_max = az_max_deg % 360.0
    if az_max <= az_min:
        az_max += 360.0
    return az_min, az_max


def _unwrap_deg(az_deg: float, az_min: float) -> float:
    """Lleva un azimut al espacio desenrollado de un recorte: con az_min=340,
    el 350 se queda en 350 y el 10 pasa a 370."""
    return az_deg + 360.0 if az_deg < az_min else az_deg


def _crop_profile(az: np.ndarray, elev: np.ndarray, trunc: np.ndarray,
                  az_min: float, az_max_u: float, step_deg: float):
    """Rota los arrays para que empiecen en az_min (quedan monótonos en
    espacio desenrollado) y recorta al sector.

    Rotar en vez de enmascarar es lo que permite que _runs, los polígonos y
    las bandas de void sigan funcionando sin enterarse del wraparound: un
    tramo de terreno que cruza el norte, hoy partido en dos por el borde del
    array, pasa a ser contiguo.
    """
    start = int(np.searchsorted(az, az_min))
    az_u = np.concatenate([az[start:], az[:start] + 360.0])
    elev = np.concatenate([elev[start:], elev[:start]])
    trunc = np.concatenate([trunc[start:], trunc[:start]])
    # margen de una muestra a cada lado para que la silueta llegue pegada al
    # borde del marco (x_at ya clampa lo que se salga)
    keep = (az_u >= az_min - step_deg) & (az_u <= az_max_u + step_deg)
    return az_u[keep], elev[keep], trunc[keep]


def _azimuth_ticks(az_min: float, az_max_u: float,
                   cropped: bool) -> list[tuple[float, str | None]]:
    """Marcas del eje X en espacio desenrollado, con su etiqueta (o None).

    Los cardinales llevan siempre su letra, comprobando también su copia +360
    para que el norte aparezca como "N (0°)" en un recorte tipo 340→400. Los
    ticks no cardinales solo se etiquetan CUANDO HAY RECORTE: así el panorama
    completo conserva exactamente su aspecto anterior (solo N/E/S/O) y el
    sector ampliado gana la escala en grados que necesita.
    """
    span = az_max_u - az_min
    for limit, step in ((15.0, 2.0), (30.0, 5.0), (90.0, 10.0),
                        (180.0, 20.0)):
        if span <= limit:
            break
    else:
        step = 45.0

    cardinals = {az: name for az, name in _CARDINALS}
    ticks: list[tuple[float, str | None]] = []
    first = math.ceil(az_min / step) * step
    tick = first
    while tick <= az_max_u + 1e-9:
        wrapped = tick % 360.0
        # sin recorte el tick final se rotula "N (360°)" (cierra el círculo,
        # como venía haciéndose); en un sector desenrollado se rotula "N (0°)"
        # para que encaje con los ticks siguientes, que ya van en 10°, 20°...
        shown = wrapped if cropped else tick
        if wrapped in cardinals:
            label = f"{cardinals[wrapped]} ({shown:.0f}°)"
        elif cropped:
            label = f"{shown:.0f}°"
        else:
            label = None
        ticks.append((tick, label))
        tick += step
    return ticks


def _place_labels(img: Image.Image, draw: ImageDraw.ImageDraw,
                  peaks: list[PeakSighting], x_at, y_at,
                  band_top: int, band_bottom: int,
                  y_plot: tuple[int, int], font: ImageFont.FreeTypeFont,
                  s: int, az_min: float, az_max_u: float) -> LabelReport:
    """Colocación voraz por intervalos: los picos, en orden de prioridad,
    reclaman una ranura horizontal; quien choca con una ya ocupada (siempre
    de mayor prioridad) se descarta entero. Sin desplazamientos: la x de la
    etiqueta ES el azimut, que es lo que permite cruzar el PNG con la foto.

    Prioridad: elevación aparente descendente (lo que domina la silueta),
    altitud como desempate, distancia después. Sin `prominence` (que
    CLAUDE.md prohíbe usar), la elevación aparente es el mejor proxy de
    relevancia visual.
    """
    # ranura medida sobre la fuente real, no constantizada: el footprint
    # horizontal de un texto rotado 90° es su altura de línea
    ascent, descent = font.getmetrics()
    slot_w = ascent + descent + 6 * s  # 6 px (a 1x: 3) de separación fija
    max_text_px = band_bottom - band_top - 6 * s

    ordered = sorted(peaks, key=lambda p: (-p.elevation_deg, -p.h_m,
                                           p.distance_m))
    occupied: list[tuple[float, float]] = []
    placed, discarded, out_of_range = [], [], []
    for sight in ordered:
        name = sight.peak.label
        # el filtro va ANTES de competir por ranura: en un sector estrecho
        # solo compiten los picos del sector, que es lo que se busca al ampliar
        az_u = _unwrap_deg(sight.azimuth_deg, az_min)
        if not az_min <= az_u <= az_max_u:
            out_of_range.append(name)
            continue
        x = x_at(az_u)
        lo, hi = x - slot_w / 2, x + slot_w / 2
        if any(not (hi <= a or lo >= b) for a, b in occupied):
            discarded.append(name)
            continue
        occupied.append((lo, hi))
        placed.append(name)

        unknown = sight.visibility is Visibility.UNKNOWN
        color = _TRUNCATED_EDGE if unknown else _INK
        text = _label_text(draw, name, sight.h_m, unknown, font, max_text_px)

        # guía desde el punto del pico en la silueta hasta la banda;
        # discontinua para UNKNOWN, coherente con los sectores ámbar
        y_anchor = min(max(y_at(sight.elevation_deg), y_plot[0]), y_plot[1])
        if unknown:
            _dashed_line(draw, x, band_bottom + 2 * s, x, y_anchor,
                         dash=4 * s, gap=3 * s, fill=_TRUNCATED_EDGE, width=s)
        else:
            draw.line([x, band_bottom + 2 * s, x, y_anchor],
                      fill=_INK_MUTED, width=s)

        _draw_rotated_label(img, text, font, color,
                            int(x), band_bottom, s)
    return LabelReport(placed=placed, discarded=discarded,
                       out_of_range=out_of_range)


def _label_text(draw: ImageDraw.ImageDraw, name: str, h_m: float,
                unknown: bool, font: ImageFont.FreeTypeFont,
                max_px: int) -> str:
    """Nombre + altitud (con '?' si UNKNOWN), truncando el NOMBRE por el
    final si no cabe: "Cerro del Collado de la Torrec…" sigue siendo
    reconocible en un mapa; cortar por el principio no. Y un nombre truncado
    hasta lo irreconocible es mejor que no dibujarlo: la posición ya es
    información útil."""
    suffix = f" {h_m:.0f} m" + ("?" if unknown else "")
    text = name + suffix
    while draw.textlength(text, font=font) > max_px and len(name) > 1:
        name = name[:-1]
        text = name.rstrip() + "…" + suffix
    return text


def _draw_rotated_label(img: Image.Image, text: str,
                        font: ImageFont.FreeTypeFont,
                        color: tuple[int, int, int], x: int, y_bottom: int,
                        s: int) -> None:
    """Texto rotado 90° (se lee de abajo arriba), con su base apoyada en
    y_bottom y centrado en x. pillow no rota texto al dibujar: se pinta en
    una capa RGBA y se rota la capa."""
    probe = ImageDraw.Draw(img)
    text_w = int(probe.textlength(text, font=font)) + 2 * s
    ascent, descent = font.getmetrics()
    text_h = ascent + descent
    layer = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((s, 0), text, font=font, fill=color + (255,))
    rotated = layer.rotate(90, expand=True)  # ancho text_h, alto text_w
    img.paste(rotated, (x - text_h // 2, y_bottom - text_w), rotated)


def _legend(draw: ImageDraw.ImageDraw, x: int, y: int,
            font: ImageFont.FreeTypeFont, scale: int,
            with_unknown_peaks: bool = False) -> None:
    entries = [
        (_TERRAIN, None, "terreno"),
        (_TRUNCATED, None, "truncado (cobertura DEM incompleta)"),
        (_NODATA_BAND, _NODATA_HATCH, "sin dato (void)"),
    ]
    if with_unknown_peaks:
        entries.append((_TRUNCATED_EDGE, None, "pico sin confirmar (?)"))
    sw = 14 * scale  # lado del cuadrado de muestra
    for fill, hatch, label in entries:
        draw.rectangle([x, y, x + sw, y + sw], fill=fill,
                       outline=_INK_MUTED, width=1 * scale)
        if hatch is not None:
            draw.line([x, y + sw, x + sw, y], fill=hatch, width=scale)
            draw.line([x, y + sw // 2, x + sw // 2, y], fill=hatch, width=scale)
            draw.line([x + sw // 2, y + sw, x + sw, y + sw // 2],
                      fill=hatch, width=scale)
        draw.text((x + sw + 6 * scale, y + sw // 2), label, font=font,
                  fill=_INK, anchor="lm")
        x += sw + 6 * scale + int(draw.textlength(label, font=font)) + 22 * scale


def _runs(classes: np.ndarray, cls: int) -> list[tuple[int, int]]:
    """Tramos contiguos [start, end) donde classes == cls."""
    mask = np.concatenate(([False], classes == cls, [False]))
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def _hatch(img: Image.Image, bx0: int, by0: int, bx1: int, by1: int,
           spacing: int, color: tuple[int, int, int], line_w: int) -> None:
    """Trama diagonal a 45° recortada a la banda (pillow no tiene clipping:
    se dibuja en una capa del tamaño exacto y se pega)."""
    w, h = bx1 - bx0, by1 - by0
    if w <= 0 or h <= 0:
        return
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    for cx in range(-h, w, spacing):
        layer_draw.line([cx, h, cx + h, 0], fill=color + (255,), width=line_w)
    img.paste(layer, (bx0, by0), layer)


def _dashed_line(draw: ImageDraw.ImageDraw, xa: float, ya: float,
                 xb: float, yb: float, dash: int, gap: int,
                 fill: tuple[int, int, int], width: int) -> None:
    length = math.hypot(xb - xa, yb - ya)
    if length == 0:
        return
    ux, uy = (xb - xa) / length, (yb - ya) / length
    d = 0.0
    while d < length:
        end = min(d + dash, length)
        draw.line([xa + ux * d, ya + uy * d, xa + ux * end, ya + uy * end],
                  fill=fill, width=width)
        d = end + gap


def _nice_ticks(lo: float, hi: float, target: int = 7) -> list[float]:
    span = hi - lo
    for step in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 15.0, 30.0):
        if span / step <= target:
            break
    first = math.ceil(lo / step) * step
    ticks = []
    t = first
    while t <= hi + 1e-9:
        ticks.append(round(t, 6))
        t += step
    return ticks


def _fmt_deg(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{text}°"


def _font(size_px: int) -> ImageFont.FreeTypeFont:
    for name in ("segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size_px)
        except OSError:
            continue
    return ImageFont.load_default(size_px)
