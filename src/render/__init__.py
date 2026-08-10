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
    """Qué picos consiguieron etiqueta y cuáles se descartaron por colisión,
    ambos en orden de prioridad. Para verificar el criterio de descarte."""
    placed: list[str]
    discarded: list[str]


def render_horizon_png(profile: HorizonProfile, path: str,
                       peaks: list[PeakSighting] | None = None,
                       width_px: int = 1600,
                       height_px: int = 520) -> LabelReport | None:
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
    """
    s = 2  # supersampling: se dibuja a 2x y se reduce con LANCZOS
    width, height = width_px * s, height_px * s
    margin_l, margin_r = 64 * s, 20 * s
    margin_t = 56 * s + (_LABEL_BAND_1X * s if peaks else 0)
    margin_b = 44 * s
    x0, x1 = margin_l, width - margin_r
    y0, y1 = margin_t, height - margin_b
    plot_w, plot_h = x1 - x0, y1 - y0

    az = np.asarray(profile.azimuths_deg, dtype=float)
    elev = np.asarray(profile.elevations_deg, dtype=float)
    trunc = np.asarray(profile.truncated_at_m, dtype=float)
    az_step_deg = az[1] - az[0] if az.size > 1 else 0.2

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
        return x0 + az_deg / 360.0 * plot_w

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
    for az_deg in range(0, 361, 45):
        gx = x_at(az_deg)
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

    for az_deg, label in _CARDINALS:
        tx = x_at(az_deg)
        draw.line([tx, y1, tx, y1 + 5 * s], fill=_INK_MUTED, width=s)
        draw.text((tx, y1 + 8 * s), f"{label} ({az_deg:.0f}°)",
                  font=font_cardinal, fill=_INK, anchor="ma")

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
                               y_plot=(y0, y1), font=font, s=s)

    img = img.resize((width_px, height_px), Image.LANCZOS)
    img.save(path, "PNG")
    return report


def _place_labels(img: Image.Image, draw: ImageDraw.ImageDraw,
                  peaks: list[PeakSighting], x_at, y_at,
                  band_top: int, band_bottom: int,
                  y_plot: tuple[int, int], font: ImageFont.FreeTypeFont,
                  s: int) -> LabelReport:
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
    placed, discarded = [], []
    for sight in ordered:
        name = sight.peak.name or f"osm:{sight.peak.osm_id}"
        x = x_at(sight.azimuth_deg)
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
    return LabelReport(placed=placed, discarded=discarded)


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
