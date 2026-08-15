"""Skyline como CAMINO GLOBAL, por programación dinámica.

El skyline es un camino continuo de la primera a la última columna. Aquí se
busca el de coste mínimo global con penalización por salto, de modo que un
atajo cresta→nube→cresta paga los dos saltos aunque su coste local sea menor.

=============================================================================
NO USAR ESTA DP SOBRE EVIDENCIA DE COLOR. MEDIDO Y DESCARTADO.
=============================================================================

Se probó exactamente eso — la DP sobre el mapa de evidencia de color de abajo,
como sustituta del detector heurístico — y **empeoró las CINCO fotos de
referencia**, medido con `scripts/eval_skyline.py` contra los alineamientos
manuales (residuo mediano contra la línea proyectada):

    foto                    heurística      DP
    141720.jpg                   1.5 px   906.2 px
    141721.jpg                   2.2 px   827.7 px
    20260812_104438.jpg          2.2 px   368.1 px
    Nerja                        1.9 px     7.5 px
    sierra.jpg                  31.3 px  1970.2 px

Y en la foto que motivó todo (`IMG_20240210_122140.jpg`, cielo cubierto) la DP
tampoco la arregla: da un camino continuo —la dispersión baja de 1070 a 550 px
y la cobertura sube de 78% a 96%— pero ese camino va POR ENCIMA DE LAS NUBES.

La lección, que es el motivo de este aviso: **el problema no era la
discontinuidad sino la evidencia.** Bajo un criterio de color, una nube es tan
"no-cielo" como una montaña: menos azul, más gris, borde marcado. La DP no
puede arreglar eso, solo vuelve COHERENTE una respuesta equivocada — y una
línea suave y falsa es más creíble, y por tanto peor, que un desastre
visiblemente disperso.

La DP sigue aquí porque es correcta y útil ENCIMA DE UNA EVIDENCIA QUE SEPA
QUÉ ES UNA NUBE (segmentación con modelo). El resto del módulo es
independiente del origen de `s(r,c)`: quien la produzca, el camino no cambia.

Y así ha sido: con la evidencia del modelo (`segmentation.py`) la misma DP
lleva la foto de las nubes a 100% de cobertura y 231.9 px de dispersión,
frente a 77.6% y 1070.3 px de la heurística, y baja el residuo de sierra.jpg
de 31.3 a 26.2 px. Mismo algoritmo, evidencia distinta: la moraleja de arriba
en positivo.
"""

import math

import numpy as np

JUMP_LIMIT_PX = 25      # salto máximo entre columnas vecinas, en filas
# λ ADIMENSIONAL: fracción del rango de coste de una columna que cuesta un
# salto del tamaño máximo. Fijarlo en unidades absolutas es un error de
# escala fácil de cometer y difícil de ver — con λ=0.6 por fila, y un rango
# de coste por columna de 0.73, moverse UNA fila costaba casi todo el rango
# y el camino óptimo salía perfectamente plano.
SMOOTHNESS = 0.35
EVIDENCE_SCALE = 12.0   # k de la sigmoide: cuán tajante es "esto no es cielo"
EVIDENCE_OFFSET = 4.0   # t: desviación (en sigmas del propio cielo) que ya
                        # no es cielo. Va en sigmas, no en niveles de gris.
MARGIN_PX = 40.0        # "lejos" al medir la fiabilidad de una columna


def sky_evidence(channels, sky_rows: int, alpha: float = 0.02) -> np.ndarray:
    """Mapa `s(r,c) ∈ [0,1]` por COLOR: probabilidad de que el píxel sea cielo.

    OJO: esta evidencia no distingue nube de terreno — ver el aviso de la
    cabecera del módulo. Se conserva para el detector heurístico y para
    comparar, no para alimentar la DP en producción.

    Mismo barrido descendente adaptativo que la heurística, con una
    diferencia: aquella BLOQUEA la columna al encontrar la cresta, y aquí hay
    que recorrerla entera. La referencia de cielo se congela en cuanto el
    píxel deja de parecerlo y se sigue comparando hacia abajo, de modo que el
    terreno acumula desviación en vez de arrastrar la referencia con él.
    """
    brightness, blueness, saturation = channels
    height, width = brightness.shape

    seed = slice(0, sky_rows)
    ref_b = np.median(brightness[seed], axis=0)
    ref_u = np.median(blueness[seed], axis=0)
    ref_s = np.median(saturation[seed], axis=0)
    dev_b = np.maximum(1.4826 * np.median(
        np.abs(brightness[seed] - ref_b), axis=0), 2.0)
    dev_u = np.maximum(1.4826 * np.median(
        np.abs(blueness[seed] - ref_u), axis=0), 2.0)
    dev_s = np.maximum(1.4826 * np.median(
        np.abs(saturation[seed] - ref_s), axis=0), 2.0)

    deviation = np.zeros((height, width), dtype=float)
    still_sky = np.ones(width, dtype=bool)
    for row in range(height):
        b, u, s = brightness[row], blueness[row], saturation[row]
        # desviación en unidades de "ruidos del propio cielo": mide cuánto se
        # aleja el píxel de lo que este cielo es en esta columna
        gap = np.maximum.reduce([
            (ref_b - b) / np.maximum(dev_b, 1.0),
            (ref_u - u) / np.maximum(dev_u, 1.0),
            (s - ref_s) / np.maximum(dev_s, 1.0),
        ])
        deviation[row] = np.maximum(gap, 0.0)

        if row < sky_rows:
            continue
        # la referencia solo se arrastra mientras el píxel siga siendo cielo;
        # en cuanto deja de serlo se congela, para que el terreno no la
        # empuje consigo y acabe pareciendo cielo oscuro
        looks_sky = gap < 3.0
        updating = still_sky & looks_sky
        still_sky &= looks_sky
        res_b, res_u, res_s = b - ref_b, u - ref_u, s - ref_s
        ref_b = np.where(updating, ref_b + alpha * res_b, ref_b)
        ref_u = np.where(updating, ref_u + alpha * res_u, ref_u)
        ref_s = np.where(updating, ref_s + alpha * res_s, ref_s)
        dev_b = np.where(updating, (1 - alpha) * dev_b + alpha * np.abs(res_b),
                         dev_b)
        dev_u = np.where(updating, (1 - alpha) * dev_u + alpha * np.abs(res_u),
                         dev_u)
        dev_s = np.where(updating, (1 - alpha) * dev_s + alpha * np.abs(res_s),
                         dev_s)

    # sigmoide: 1 = cielo seguro, 0 = terreno seguro. El exponente se acota
    # para no desbordar con desviaciones enormes (terreno muy oscuro).
    z = np.clip(EVIDENCE_SCALE * (deviation - EVIDENCE_OFFSET)
                / EVIDENCE_OFFSET, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))


def boundary_cost(sky: np.ndarray) -> np.ndarray:
    """Coste de que la frontera pase por cada `(r,c)`.

    Cielo mal explicado ARRIBA más terreno mal explicado ABAJO:

        coste(r,c) = media(1−s por encima de r) + media(s por debajo de r)

    Con sumas acumuladas sale para todas las filas de golpe, O(H·W). Es lo
    que convierte "dónde está el borde" en "qué partición cielo/terreno
    explica mejor la columna entera", que es una pregunta mucho más robusta
    frente a un borde de nube.
    """
    sky = np.asarray(sky, dtype=float)
    height = sky.shape[0]
    rows = np.arange(height, dtype=float)[:, None]

    cum_sky = np.cumsum(sky, axis=0)          # cielo acumulado hasta r-1
    above_sky = np.vstack([np.zeros((1, sky.shape[1])), cum_sky[:-1]])
    total_sky = cum_sky[-1][None, :]

    # arriba deberían ser todo cielo: penaliza el (1-s) acumulado
    above = np.divide(rows - above_sky, np.maximum(rows, 1.0))
    # abajo, todo terreno: penaliza el s que quede
    below_count = np.maximum(height - 1 - rows, 1.0)
    below = np.divide(total_sky - above_sky - sky, below_count)
    return above + below


def edge_cost(small: np.ndarray) -> np.ndarray:
    """Coste BAJO donde hay un borde horizontal fuerte, en `[0, 1]`.

    Existe porque el modelo y la imagen saben cosas distintas. El modelo dice
    QUÉ es cielo pero saca los logits a 1/4 de su entrada: con 1536 de lado
    largo el cuanto de la máscara son ~10 px de la foto, y por debajo de eso
    no puede situar nada — medido, el residuo sigue al cuanto (8.4 px a 512,
    2.0 px a 1536). El gradiente de la imagen no sabe qué es una nube, pero
    está a resolución de trabajo. Uno para acertar de zona, el otro de fila.

    Se normaliza por el percentil 95 DE CADA COLUMNA: si no, una columna de
    bajo contraste (calima, contraluz) tendría bordes débiles en términos
    absolutos y quedaría descartada entera frente a otra bien iluminada.
    """
    small = np.asarray(small, dtype=float)
    luminance = small @ np.array([0.299, 0.587, 0.114])
    gradient = np.zeros_like(luminance)
    gradient[1:-1] = np.abs(luminance[2:] - luminance[:-2])
    reference = np.percentile(gradient, 95, axis=0, keepdims=True)
    return 1.0 - np.clip(gradient / np.maximum(reference, 1e-6), 0.0, 1.0)


def best_path(cost: np.ndarray, jump_limit: int = JUMP_LIMIT_PX,
              smoothness: float = SMOOTHNESS,
              ) -> tuple[np.ndarray, np.ndarray]:
    """Camino de coste mínimo columna a columna, con penalización de salto.

        mejor(c,r) = coste(r,c) + min_{|r'−r| ≤ K} [ mejor(c−1,r') + λ·|r−r'| ]

    Devuelve (filas del camino, margen por columna). El margen es cuánto peor
    es el mejor camino que pasa LEJOS de la solución en esa columna: mide si
    la evidencia manda ahí o si el camino pasa por inercia de sus vecinos.

    Con `smoothness=0` degenera en elegir el mínimo de cada columna por
    separado, que es justo el comportamiento que se quiere superar.
    """
    cost = np.asarray(cost, dtype=float)
    height, width = cost.shape
    offsets = np.arange(-jump_limit, jump_limit + 1)

    # λ en unidades del propio coste: un salto del tamaño máximo cuesta
    # `smoothness` veces el rango típico de una columna
    scale = float(np.median(cost.max(axis=0) - cost.min(axis=0)))
    lam = smoothness * max(scale, 1e-9) / max(jump_limit, 1)
    penalty = lam * np.abs(offsets).astype(float)

    def sweep(sequence):
        """Coste acumulado y elección, recorriendo las columnas en ese orden."""
        acc = np.empty((width, height), dtype=float)
        pick_from = np.empty((width, height), dtype=np.int32)
        first = sequence[0]
        acc[first] = cost[:, first]
        pick_from[first] = np.arange(height)
        for col in sequence[1:]:
            previous = acc[col - 1] if sequence[1] > sequence[0] else acc[col + 1]
            shifted = np.full((offsets.size, height), np.inf)
            for i, off in enumerate(offsets):
                lo, hi = max(0, -off), min(height, height - off)
                if lo < hi:
                    shifted[i, lo:hi] = previous[lo + off:hi + off] + penalty[i]
            best = np.argmin(shifted, axis=0)
            acc[col] = cost[:, col] + shifted[best, np.arange(height)]
            pick_from[col] = np.arange(height) + offsets[best]
        return acc, pick_from

    forward, choice = sweep(list(range(width)))
    backward, _ = sweep(list(range(width - 1, -1, -1)))

    path = np.empty(width, dtype=np.int64)
    path[-1] = int(np.argmin(forward[-1]))
    for col in range(width - 1, 0, -1):
        path[col - 1] = choice[col][path[col]]

    # MARGEN EXACTO: coste del mejor camino COMPLETO que pasa por (col, r),
    # que es forward + backward menos el coste de la casilla, contado dos
    # veces. Con solo el barrido de ida el margen mediría la mitad del
    # problema y daría por fiables columnas que van por inercia de sus
    # vecinas. Normalizado por la escala para que el umbral sea comparable
    # entre fotos.
    through = forward + backward - cost.T
    optimum = through.min(axis=1)
    rows = np.arange(height)
    margin = np.empty(width, dtype=float)
    for col in range(width):
        far = np.abs(rows - path[col]) > MARGIN_PX
        margin[col] = (float(through[col][far].min()) - optimum[col]
                       if far.any() else 0.0)
    return path, margin / max(scale, 1e-9)


DEFAULT_DETECTOR = "modelo+dp"


def build_detector(nombre: str = DEFAULT_DETECTOR, verbose: bool = False):
    """Devuelve el detector pedido, con la firma
    `(photo_np) -> (columnas_px, filas_px, validas)`.

    - `heuristica`: el de search.py. Referencia histórica, y RESPALDO
      automático cuando el modelo no está disponible.
    - `modelo`: segmentación de cielo con red; frontera por columna.
    - `modelo+dp`: la misma segmentación con el camino global encima. Es el
      DEFECTO, pese a depender de una dependencia opcional: gana en cobertura
      en las seis fotos de prueba, arregla la de cielo cubierto (donde la
      heurística da 1070 px de dispersión y el modelo 232) y empata en error
      de azimut. El respaldo hace que esa elección no cueste nada a quien no
      tenga onnxruntime.

    NO existe `dp` a secas sobre evidencia de color: medido, empeora las
    cinco fotos de referencia (ver cabecera del módulo).
    """
    from src.align.search import detect_photo_skyline

    if nombre == "heuristica":
        return detect_photo_skyline
    if nombre in ("modelo", "modelo+dp"):
        from src.align.segmentation import build_model_detector
        return build_model_detector(use_dp=(nombre == "modelo+dp"),
                                    verbose=verbose)
    raise ValueError(f"detector desconocido: {nombre!r}")


def detect_skyline_dp(photo_np: np.ndarray, max_width: int = 1200,
                      min_margin: float = 0.02,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cresta por camino global. Misma firma que `detect_photo_skyline`.

    `valid` cambia de significado: la DP SIEMPRE devuelve un camino completo,
    así que ya no quiere decir "encontré cresta" sino "aquí el camino es
    fiable". Sin ese matiz la cobertura saldría del 100% siempre, basura
    incluida, y se perdería la señal que avisa de que una foto no sirve.
    """
    photo = np.asarray(photo_np, dtype=float)
    width = photo.shape[1]
    step = max(1, math.ceil(width / max_width))
    small = photo[::step, ::step]
    small_h, small_w = small.shape[0], small.shape[1]

    channels = (small.mean(axis=2),
                small[..., 2] - small[..., 0],
                small.max(axis=2) - small.min(axis=2))
    sky_rows = max(4, small_h // 20)

    sky = sky_evidence(channels, sky_rows)
    cost = boundary_cost(sky)
    jump = max(4, min(JUMP_LIMIT_PX, small_h // 12))
    path, margin = best_path(cost, jump_limit=jump)

    valid = margin >= min_margin
    if valid.sum() >= 5:
        from src.align.search import _drop_short_segments
        valid = _drop_short_segments(path.astype(float), valid, small_h,
                                     small_w)

    columns_px = np.arange(small_w) * step + step / 2.0
    rows_px = path.astype(float) * step + step / 2.0
    return columns_px, rows_px, valid
