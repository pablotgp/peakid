"""Dibuja la cresta DETECTADA sobre la foto. Sin alinear ni proyectar nada.

    python scripts/dump_skyline.py FOTO.jpg [--detector modelo+dp] [--out X.png]

Aísla el detector de todo lo demás. Cuando el alineamiento no cuadra, la
pregunta "¿está mal el detector o están mal los parámetros?" no se puede
responder mirando la línea proyectada, porque ahí se mezclan las dos cosas.
Aquí no hay perfil, ni DEM, ni cámara: solo lo que el detector cree que es el
borde del cielo.

No es un test: depende de fotos que no están versionadas.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.align import load_oriented_photo                    # noqa: E402
from src.align.skyline import DEFAULT_DETECTOR, build_detector  # noqa: E402

_VALIDO = (255, 59, 48)      # mismo rojo que la silueta fiable de la GUI
_DUDOSO = (255, 149, 0)      # mismo ámbar que el "sin dato" de render/
_GROSOR = 2


def dibuja(photo_np: np.ndarray, columns_px, rows_px, valid,
           max_width: int = 1600) -> Image.Image:
    """Foto con la cresta encima: rojo donde el detector se fía, ámbar donde
    no. Las columnas descartadas SE DIBUJAN igualmente: saber dónde falla el
    detector es la mitad de la información, y ocultarlas hace que un detector
    con 30% de cobertura parezca impecable."""
    photo = np.asarray(photo_np, dtype=np.uint8)
    height, width = photo.shape[0], photo.shape[1]
    step = max(1, -(-width // max_width))
    lienzo = photo[::step, ::step].copy()
    alto, ancho = lienzo.shape[0], lienzo.shape[1]

    for c_px, r_px, ok in zip(np.asarray(columns_px), np.asarray(rows_px),
                              np.asarray(valid)):
        if not np.isfinite(r_px):
            continue
        c, r = int(round(c_px / step)), int(round(r_px / step))
        if not 0 <= c < ancho:
            continue
        color = _VALIDO if ok else _DUDOSO
        for dr in range(-_GROSOR, _GROSOR + 1):
            if 0 <= r + dr < alto:
                lienzo[r + dr, c] = color
    return Image.fromarray(lienzo)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("photo")
    p.add_argument("--detector", default=DEFAULT_DETECTOR,
                   choices=("heuristica", "modelo", "modelo+dp"))
    p.add_argument("--out", default=None,
                   help="PNG de salida (defecto: FOTO.skyline.DETECTOR.png)")
    p.add_argument("--max-width", type=int, default=1600)
    args = p.parse_args()

    img, photo_np = load_oriented_photo(args.photo)
    print(f"{Path(args.photo).name}  {img.size[0]}x{img.size[1]}"
          f"  (orientación EXIF aplicada)")

    detector = build_detector(args.detector, verbose=True)
    columns_px, rows_px, valid = detector(photo_np)

    n = int(np.sum(valid))
    print(f"detector '{args.detector}': {n}/{len(valid)} columnas válidas "
          f"({np.mean(valid):.0%})")
    if n:
        filas = np.asarray(rows_px)[np.asarray(valid)]
        print(f"  filas: mediana {np.median(filas):.0f}, "
              f"desv. típica {filas.std():.1f}, "
              f"rango {filas.min():.0f}-{filas.max():.0f} "
              f"de {img.size[1]} px de alto")

    out = Path(args.out) if args.out else Path(args.photo).with_suffix("")
    if args.out is None:
        out = out.with_name(f"{out.name}.skyline."
                            f"{args.detector.replace('+', '_')}.png")
    dibuja(photo_np, columns_px, rows_px, valid, args.max_width).save(out)
    print(f"escrito {out}   (rojo = columna válida, ámbar = descartada)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
