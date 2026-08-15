"""Segmentación de cielo con modelo, vía ONNX Runtime.

Por qué un modelo y no más heurística: bajo un criterio de color una nube es
tan "no-cielo" como una montaña, y ninguna combinación de umbrales lo arregla
(ver el aviso medido en `skyline.py`). Hace falta algo que sepa qué es una
nube.

DEPENDENCIA OPCIONAL. Sin `onnxruntime` o sin el fichero del modelo, esto no
revienta: `build_model_detector` devuelve el detector heurístico y lo dice.
Igual que `peaks/` sigue funcionando sin red.

`torch` NUNCA es dependencia de ejecución: el `.onnx` se obtiene una vez
(descarga o exportación en un entorno desechable) y aquí solo se ejecuta.

Modelo: SegFormer-B0 finetuneado en ADE20K, cuya clase **2 es `sky`**. 15.3 MB.
El `.onnx` vive en `models/` y NO se versiona, igual que los tiles `.hgt`.
"""

import math
from pathlib import Path

import numpy as np

MODEL_PATH = Path("models/segformer_b0_ade.onnx")
SKY_CLASS = 2            # índice de "sky" en ADE20K

# El preprocesador del modelo pide 512x512, pero el grafo ONNX tiene los ejes
# de alto y ancho DINÁMICOS y SegFormer es tolerante a la escala, así que se
# puede pedir más. Importa mucho: el modelo saca los logits a 1/4 de la
# entrada, y ese cuanto es el suelo del error de localización. Medido, el
# residuo mediano contra los alineamientos manuales sigue al cuanto:
#
#     entrada    cuanto (px de foto)   141720   141721   Nerja   sierra
#     512                      ~32       8.4      4.8     3.2     29.9
#     1024                     ~16       5.2      3.2     2.7     27.7
#     1536                     ~11       2.0      3.0     2.6     27.1
#     2048                      ~8       2.2      2.8     2.4     27.2
#
# 1536 es donde se estanca: 2048 no mejora y triplica el tiempo (de ~6 s a
# ~25 s por foto). Se conserva la RELACIÓN DE ASPECTO en vez de cuadrar la
# imagen; cuadrarla desperdicia resolución en el eje largo.
INPUT_LONG_SIDE = 1536
SIZE_MULTIPLE = 32       # SegFormer reduce por 32; sin esto la salida cojea
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EDGE_WEIGHT = 0.3        # peso del término de borde frente al de región

_session = None


class ModelUnavailable(RuntimeError):
    """No hay modelo utilizable: falta onnxruntime o falta el .onnx."""


def load_session(model_path: Path = MODEL_PATH):
    """Sesión de inferencia, cacheada. Lanza ModelUnavailable si no se puede.

    El import de onnxruntime va DENTRO, no en la cabecera: el módulo tiene
    que poder importarse en una instalación sin la dependencia opcional.
    """
    global _session
    if _session is not None:
        return _session
    try:
        import onnxruntime as ort
    except Exception as err:      # ImportError, y también fallos de DLL
        raise ModelUnavailable(f"onnxruntime no disponible: {err}") from err
    if not model_path.exists():
        raise ModelUnavailable(f"falta el modelo {model_path}")
    _session = ort.InferenceSession(str(model_path),
                                    providers=["CPUExecutionProvider"])
    return _session


def _input_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Tamaño de entrada que conserva el aspecto, en múltiplos de 32."""
    if width >= height:
        target_w, target_h = long_side, max(1, round(long_side * height / width))
    else:
        target_h, target_w = long_side, max(1, round(long_side * width / height))
    snap = lambda v: max(SIZE_MULTIPLE, (v // SIZE_MULTIPLE) * SIZE_MULTIPLE)
    return snap(target_w), snap(target_h)


def sky_probability(photo_np: np.ndarray, model_path: Path = MODEL_PATH,
                    long_side: int = INPUT_LONG_SIDE) -> np.ndarray:
    """Mapa `s(r,c) ∈ [0,1]` de probabilidad de cielo, al tamaño de la foto
    submuestreada que se le pase."""
    from PIL import Image

    session = load_session(model_path)
    photo = np.asarray(photo_np)
    height, width = photo.shape[0], photo.shape[1]
    target_w, target_h = _input_size(width, height, long_side)

    small = Image.fromarray(photo.astype(np.uint8)).resize(
        (target_w, target_h), Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    name = session.get_inputs()[0].name
    logits = session.run(None, {name: x})[0][0]     # (150, h, w)
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    sky = exp[SKY_CLASS] / exp.sum(axis=0)

    # de la rejilla del modelo (1/4 de la entrada) al tamaño de trabajo
    return np.asarray(
        Image.fromarray((sky * 255.0).astype(np.uint8)).resize(
            (width, height), Image.BILINEAR), dtype=float) / 255.0


def _fallback_notice_path() -> Path:
    """Marca de 'ya se avisó una vez' en esta máquina."""
    return Path.home() / ".peakid" / "aviso_modelo_ausente"


def announce_fallback(reason: str, verbose: bool = False,
                      marker: Path | None = None) -> bool:
    """Avisa del respaldo al detector heurístico, y devuelve si avisó.

    El detector por defecto depende de una dependencia OPCIONAL, así que el
    respaldo es la ruta normal para quien no la tenga — y para esa persona la
    herramienta funciona. Un aviso en cada ejecución sería ruido sobre algo
    que no ha elegido y que no le impide nada: se dice UNA vez por máquina
    (para que sepa que existe la opción y cómo activarla) y luego se calla,
    salvo `--verbose`.
    """
    marker = marker or _fallback_notice_path()
    if not verbose:
        try:
            if marker.exists():
                return False
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(reason + "\n", encoding="utf-8")
        except OSError:
            # sin sitio donde marcar, callar: preferible a avisar siempre
            return False
    print(f"nota: {reason}.\n"
          f"      Se usa el detector heurístico de color, que funciona pero "
          f"falla con cielo cubierto.\n"
          f"      Para el detector con modelo: pip install onnxruntime y el "
          f"fichero {MODEL_PATH}.\n"
          f"      (esto solo se dice una vez; con --verbose, siempre)")
    return True


def build_model_detector(use_dp: bool = True, model_path: Path = MODEL_PATH,
                         max_width: int = 1200, verbose: bool = False):
    """Detector de cresta basado en el modelo. Si el modelo no está
    disponible, devuelve el heurístico — la herramienta no se queda sin
    detector por una dependencia opcional."""
    from src.align.search import detect_photo_skyline

    try:
        load_session(model_path)
    except ModelUnavailable as err:
        announce_fallback(str(err), verbose=verbose)
        return detect_photo_skyline

    def detector(photo_np: np.ndarray):
        from src.align.skyline import (JUMP_LIMIT_PX, best_path, boundary_cost,
                                       edge_cost)

        photo = np.asarray(photo_np, dtype=float)
        width = photo.shape[1]
        step = max(1, math.ceil(width / max_width))
        small = photo[::step, ::step]
        small_h, small_w = small.shape[0], small.shape[1]

        sky = sky_probability(small, model_path)
        # frontera cruda del modelo: primera fila que deja de ser cielo
        is_sky = sky >= 0.5
        found = (~is_sky).any(axis=0)
        has_sky = is_sky.any(axis=0)
        raw = np.argmax(~is_sky, axis=0).astype(float)

        if not use_dp:
            rows, valid = raw, found & has_sky
        else:
            # cuanto de la máscara EN FILAS DE TRABAJO: es la incertidumbre
            # real del modelo, y por tanto la anchura natural de la banda
            _, target_h = _input_size(small_w, small_h, INPUT_LONG_SIDE)
            quantum = small_h / max(target_h / 4.0, 1.0)
            band = int(max(4, round(2.0 * quantum)))

            cost = boundary_cost(sky)
            span = float(cost.max() - cost.min())
            cost = (cost - cost.min()) / max(span, 1e-9)
            cost = cost + EDGE_WEIGHT * edge_cost(small)

            # La banda es lo que hace segura la evidencia de borde: el borde
            # de una nube es tan fuerte como el de una cresta, así que sin
            # acotar dónde puede pasar el camino, el término de borde
            # reintroduce justo el fallo que el modelo vino a resolver.
            usable = np.flatnonzero(found & has_sky)
            if usable.size >= 2:
                center = np.interp(np.arange(small_w), usable, raw[usable])
                far = np.abs(np.arange(small_h)[:, None] - center[None, :]) > band
                cost = cost + far * 10.0

            path, margin = best_path(
                cost, jump_limit=max(4, min(JUMP_LIMIT_PX, band)))
            rows = path.astype(float)
            valid = (margin >= 0.02) & has_sky

        if valid.sum() >= 5:
            from src.align.search import _drop_short_segments
            valid = _drop_short_segments(rows, valid, small_h, small_w)

        columns_px = np.arange(small_w) * step + step / 2.0
        rows_px = rows * step + step / 2.0
        return columns_px, rows_px, valid

    return detector
