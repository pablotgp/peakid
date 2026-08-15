"""Búsqueda automática de alineamiento por fuerza bruta (asistente).

TODO LO QUE HAY AQUÍ ES UNA HEURÍSTICA DELIBERADA para asistir el etiquetado
manual, NO la solución de la fase 2. El detector de cresta de abajo es el
mismo tipo de filtro clásico que ya demostró sus límites (calima, contraluz,
primeros planos); segmentar el skyline de verdad es el objetivo del modelo de
segmentación. Esta búsqueda no sustituye el juicio del usuario: solo le deja
empezar cerca, y el resultado SIEMPRE se valida o corrige a mano en la GUI.

La búsqueda en sí es el caso 7 del contrato en su versión sintética: dado un
perfil desplazado artificialmente, recuperar el desplazamiento (±0.1°).
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.align import AlignmentParams, project_profile, projected_y_per_column

# rejilla gruesa
COARSE_AZ_HALF_DEG = 20.0
COARSE_AZ_STEP_DEG = 0.5
FOV_RANGE_DEG = (10.0, 80.0)   # incluye teleobjetivo: un 15-20° real caía
                               # fuera del rango antiguo (40-75) y la
                               # búsqueda se saturaba en el borde
FOV_GRID_RATIO = 1.05          # rejilla GEOMÉTRICA: resolución relativa
                               # constante. Con paso lineal, 2.5° es un 3%
                               # a 75° pero un 17% a 15°, justo donde más
                               # falta precisión.
# Límites FÍSICOS de la cámara, no de comodidad. El de inclinación estaba en
# 10° y es demasiado estrecho: fotografiar una cima de 2500 m desde un valle
# a 9 km exige mirar 14.5° hacia arriba (caso real, Naranjo de Bulnes desde
# la AS-264), así que la etapa gruesa recortaba el valor correcto y la fina
# se escapaba fuera del rango explorado.
PITCH_LIMIT_DEG = 30.0
# El giro NO estaba acotado en la etapa fina, y la búsqueda devolvía −30°:
# ninguna cámara en mano sale así, y ese grado de libertad falso deja que el
# ajuste se retuerza para encajar ruido y hunde a los candidatos buenos.
ROLL_LIMIT_DEG = 15.0
COARSE_ROLL_HALF_DEG = 3.0
COARSE_ROLL_STEP_DEG = 1.0  # ya no se rejilla el giro: se resuelve cerrado
COARSE_PROFILE_STRIDE = 5   # perfil a 1° en la gruesa (0.2° en la fina)

# Barrido completo (sin pista de azimut): más grueso, porque su cometido es
# localizar REGIONES candidatas y el refinado afina cada una.
SWEEP_AZ_STEP_DEG = 1.0
SWEEP_FOV_RATIO = 1.10

# LÍMITE MEDIDO DE ESTA COMPROBACIÓN, no lo olvides antes de fiarte de
# `reliable`: la detección de ambigüedad contrasta el MEJOR contra UN solo
# alternativo lejano. No cubre el caso de un CONTINUO de óptimos parecidos,
# que es justo el que se da cuando el perfil es autosemejante.
#
# Caso real (141720.jpg, Torre del Mar). Tres hipótesis separadas 25° en
# azimut, con el error en píxeles MEJORANDO al ensanchar el espacio:
#     az 26.2 -> 29.1 px      az 34.4 -> 14.2 px      az 51.2 ->  9.5 px
# y la búsqueda declaró el tercero "sin reservas": ningún alternativo lejano
# concreto lo batía por poco, pero el mínimo global estaba en el sitio
# equivocado. La comprobación geográfica lo desmiente sin lugar a dudas —
# sobre el ápice de la foto, cada candidato pone:
#     az 26.2 -> Veas 703 m (y a 198 px del ápice)
#     az 34.4 -> Mojón de tres Términos 2065 m (a 18 px), con Maroma 2069 y
#                Cima de Tejeda 2069 pegadas: el macizo dominante sobre el
#                bulto dominante  <-- el correcto
#     az 51.2 -> Benthomiz 708 m (a 14 px), y saca a La Maroma del encuadre
# Un cerro de 708 m no puede ser el macizo que domina la foto.
#
# Conclusión operativa: el error en píxeles ordena candidatos DENTRO de una
# hipótesis, no entre hipótesis. Para elegir entre hipótesis hay que mirar
# los topónimos (qué cima cae sobre qué bulto y con qué altitud), que es
# información que este módulo no usa. Por eso la búsqueda es un asistente y
# el juicio final es del usuario.
#
# Un segundo óptimo a más de esta separación en azimut es una hipótesis
# GENUINAMENTE distinta, no un vecino de rejilla.
AMBIGUITY_AZ_DEG = 10.0
# Si ese óptimo lejano no es peor que el mejor en al menos esta fracción,
# la búsqueda no distingue entre ambos y no debe presentarse como solución.
# Calibrado sobre 141720.jpg (panorama ancho, firma rica): con el espacio
# abierto de par en par (FOV 10-80) el margen mide 0.02 y el resultado ES
# genuinamente ambiguo; acotando con pistas razonables sube a 0.73. Los dos
# regímenes están separados por más de un orden de magnitud, así que el
# umbral no es delicado.
AMBIGUITY_MARGIN = 0.25


@dataclass
class SearchCandidate:
    params: AlignmentParams
    error_px: float    # desajuste vertical cresta<->línea; MAGNITUD DE RANKING
    error_deg: float   # el mismo error en grados, solo para leerlo
    coverage: float    # fracción de columnas de cresta bajo la línea proyectada
    saturated_fraction: float = 0.0   # columnas cuyo residuo toca el recorte

    @property
    def saturated(self) -> bool:
        """El error ya no mide la calidad, solo dice 'muy mal'.

        Sin esto un desajuste de 350 px se lee como 40 y parece a un paso de
        encajar. Pasó de verdad: se informó un ajuste como '38 px, casi
        bueno' cuando la línea proyectada iba a 350 px de la cresta."""
        return self.saturated_fraction >= SATURATED_FRACTION


@dataclass
class SearchResult:
    """Resultado con sus reservas: un número sin ellas engaña más que ayuda."""
    candidates: list[SearchCandidate]
    ambiguous: bool             # hay otro óptimo lejano casi igual de bueno
    ambiguity_margin: float     # cuánto peor es ese óptimo lejano (fracción)
    alternative: SearchCandidate | None
    fov_at_edge: bool           # el óptimo se apoya en el borde del rango FOV
    az_at_edge: bool            # ídem en azimut: el verdadero puede estar fuera
    az_range_deg: tuple[float, float]
    fov_range_deg: tuple[float, float]

    @property
    def reliable(self) -> bool:
        """Sin reservas DETECTABLES por este módulo.

        No es una garantía de que el resultado sea correcto: ver el límite
        medido junto a AMBIGUITY_AZ_DEG. Un mínimo global en el sitio
        equivocado puede salir "fiable" si ningún alternativo concreto lo
        roza. La comprobación que zanja es geográfica (qué cima cae sobre qué
        bulto), y esa la hace el usuario mirando las etiquetas de peaks/.
        """
        return bool(self.candidates) and not (self.ambiguous or self.fov_at_edge
                                              or self.az_at_edge)


def detect_photo_skyline(photo_np: np.ndarray, max_width: int = 1200,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cresta de la foto: primera fila no-cielo por columna, de ARRIBA ABAJO.

    Heurística asistente (ver cabecera del módulo). Criterio de cielo por
    color — más azul que rojo, claro, poco saturado — contra una referencia
    que se arrastra fila a fila mientras el píxel siga siendo cielo, de modo
    que el degradado se sigue en vez de predecirse. Persistencia mínima para
    saltarse cables y pájaros, y límite de salto entre columnas vecinas para
    descartar detecciones incoherentes.

    Devuelve (columnas_px, filas_px, confianza) en coordenadas de la foto
    COMPLETA. Las columnas sin confianza deben ignorarse; con calima densa o
    contraluz pueden ser casi todas, y eso es información, no un fallo.
    """
    photo = np.asarray(photo_np, dtype=float)
    width = photo.shape[1]
    step = max(1, math.ceil(width / max_width))
    small = photo[::step, ::step]
    small_h, small_w = small.shape[0], small.shape[1]

    channels = (small.mean(axis=2),                      # brillo
                small[..., 2] - small[..., 0],           # azul - rojo
                small.max(axis=2) - small.min(axis=2))   # saturación

    sky_rows = max(4, small_h // 20)
    min_run = max(3, small_h // 60)

    # BARRIDO DESCENDENTE ADAPTATIVO. Cualquier modelo global del cielo
    # (tendencia lineal extrapolada, en una o dos pasadas) falla en fotos
    # reales: el degradado del cielo despejado NO es lineal a lo largo de
    # miles de filas, y el detector disparaba en pleno cielo, cientos de
    # píxeles por encima de la sierra (medido: filas ~1360 con la sierra en
    # ~2100). En vez de extrapolar, la referencia de cielo se ACTUALIZA fila
    # a fila (media móvil exponencial sobre los píxeles que siguen siendo
    # cielo): el degradado se sigue, no se predice.
    rows_small, found = _adaptive_scan(channels, sky_rows, min_run)

    valid = found.copy()
    if valid.sum() >= 5:
        # límite de salto: lo que se aparta de la mediana local no es cresta
        filled = np.where(valid, rows_small, np.median(rows_small[valid]))
        smooth = _median_filter(filled, 9)
        valid &= np.abs(rows_small - smooth) <= max(4.0, small_h * 0.03)
        valid = _drop_short_segments(rows_small, valid, small_h, small_w)

    columns_px = np.arange(small_w) * step + step / 2.0
    rows_px = rows_small * step + step / 2.0
    return columns_px, rows_px, valid


def _drop_short_segments(rows: np.ndarray, valid: np.ndarray, height: int,
                         width: int) -> np.ndarray:
    """Descarta los tramos CORTOS que quedan aislados entre saltos bruscos.

    Un obstáculo de primer plano —un poste, una farola, una rama, el borde de
    una valla— rompe la cresta con un escalón de cientos de píxeles respecto
    a las columnas vecinas, y su detección envenena el ajuste: la recta de
    pitch/roll se tuerce por unas pocas columnas que no son horizonte. Aquí
    la cresta se parte en segmentos por esos escalones y se tira lo que sea
    demasiado corto para ser relieve de verdad.

    LÍMITE, y por eso existe además el recuadro manual de la GUI: esto caza
    obstáculos ESTRECHOS. Una valla que cruza media foto produce un segmento
    largo y coherente, indistinguible de una cresta real sin más contexto —
    ahí decide el usuario acotando la zona limpia.
    """
    columns = np.flatnonzero(valid)
    if columns.size < 5:
        return valid

    # umbral adaptativo a lo accidentado que sea ESTE horizonte: un escalón
    # es lo que se sale del salto típico entre columnas contiguas
    steps = np.abs(np.diff(rows[columns]))
    contiguous = np.diff(columns) == 1
    typical = float(np.median(steps[contiguous])) if contiguous.any() else 1.0
    jump = max(height * 0.02, 6.0 * max(typical, 1.0))

    # un hueco de columnas inválidas también rompe segmento
    breaks = (steps > jump) | ~contiguous
    segment = np.concatenate(([0], np.cumsum(breaks)))

    min_len = max(4, int(0.015 * width))
    keep = np.zeros(valid.shape, dtype=bool)
    for label in np.unique(segment):
        members = columns[segment == label]
        if members.size >= min_len:
            keep[members] = True
    # si el criterio se lleva casi todo, la foto es así de fragmentada y es
    # mejor conservar lo que había que quedarse sin nada
    return keep if keep.sum() >= max(min_len, 0.2 * columns.size) else valid


def _adaptive_scan(channels, sky_rows: int, min_run: int, alpha: float = 0.02,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Barrido descendente por columna con referencia de cielo que se
    actualiza sobre la marcha.

    La referencia arranca en la franja superior y, mientras el píxel siga
    pareciendo cielo, se arrastra con una media móvil exponencial. Así el
    degradado del cielo se SIGUE en vez de predecirse: es lo que distingue
    "el cielo se apaga poco a poco" (cientos de filas, la EMA lo absorbe) de
    "aquí empieza el terreno" (unas pocas filas, la EMA no llega a tiempo y
    salta el umbral). El paso vectoriza sobre columnas; el bucle es sobre
    filas porque la referencia de cada fila depende de la anterior.

    Devuelve (fila de cresta por columna, encontrada).
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

    crest = np.zeros(width, dtype=float)
    found = np.zeros(width, dtype=bool)
    run = np.zeros(width, dtype=int)
    active = np.ones(width, dtype=bool)

    for row in range(sky_rows, height):
        b, u, s = brightness[row], blueness[row], saturation[row]
        not_sky = ((b < ref_b - np.maximum(12.0, 4.0 * dev_b))
                   | (u < ref_u - np.maximum(10.0, 4.0 * dev_u))
                   | (s > ref_s + np.maximum(18.0, 4.0 * dev_s)))

        # persistencia: una racha corta es un cable o un pájaro, no la cresta
        run = np.where(active, np.where(not_sky, run + 1, 0), run)
        locking = active & (run >= min_run)
        if locking.any():
            crest[locking] = row - min_run + 1
            found |= locking
            active &= ~locking
        if not active.any():
            break

        # la referencia solo se arrastra donde el píxel sigue siendo cielo
        updating = active & ~not_sky
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
    return crest, found


def _fit_pitch_roll(residual_px: np.ndarray, offset_px: np.ndarray,
                    ) -> tuple[float, float, float] | None:
    """Recta robusta `residuo = a + b·(x − W/2)`.

    Devuelve (a, b, error_px). La ordenada `a` es el desplazamiento vertical
    (inclinación) y la pendiente `b` el giro EN RADIANES — el modelo de
    primer orden verificado en solve_pitch_roll. Las columnas que se apartan
    más de 3·MAD se descartan antes de ajustar: un tejado o un arbusto
    detectado como cresta no debe torcer la recta.
    """
    if residual_px.size < 8:
        return None
    deviation = np.abs(residual_px - np.median(residual_px))
    scale = 1.4826 * np.median(deviation)
    keep = deviation <= 3.0 * scale if scale > 0 else np.ones_like(
        residual_px, dtype=bool)
    if keep.sum() < 8:
        keep = np.ones_like(residual_px, dtype=bool)
    design = np.vstack([np.ones(int(keep.sum())), offset_px[keep]]).T
    (intercept, slope), *_ = np.linalg.lstsq(design, residual_px[keep],
                                             rcond=None)
    err_px = _clipped_mean_abs(residual_px - (intercept + slope * offset_px))
    return float(intercept), float(slope), err_px


def _refine(azimuths_deg, elevations_deg, params: AlignmentParams,
            cols, rows, width_px, height_px, min_columns, az_step, fov_span,
            fov_lo, fov_hi) -> AlignmentParams:
    """Refina un candidato en azimut × FOV, con inclinación y giro cerrados.

    Antes esta etapa rejillaba los cuatro parámetros (13×11×13×13 ≈ 24 000
    evaluaciones por candidato); resolviendo los dos verticales en forma
    cerrada bajan a ~143 y además salen más finos que el paso de 0.1°. Eso es
    lo que permite refinar CINCO candidatos en vez de solo el mejor.
    """
    best = params
    best_err = math.inf
    for az in np.arange(params.azimuth_deg - az_step,
                        params.azimuth_deg + az_step + 1e-9, az_step / 8.0):
        for fov in np.clip(np.arange(params.hfov_deg - fov_span,
                                     params.hfov_deg + fov_span + 1e-9,
                                     max(fov_span / 5.0, 0.1)),
                           fov_lo, fov_hi):
            trial = AlignmentParams(az % 360.0, float(fov),
                                    params.pitch_deg, params.roll_deg)
            solved = solve_pitch_roll(azimuths_deg, elevations_deg, trial,
                                      cols, rows, width_px, height_px,
                                      min_columns=min_columns)
            if solved is not None:
                trial = AlignmentParams(trial.azimuth_deg, trial.hfov_deg,
                                        solved[0], solved[1])
            scored = _score_exact(azimuths_deg, elevations_deg, trial, cols,
                                  rows, width_px, height_px, min_columns)
            if scored is not None and scored[0] < best_err:
                best_err, best = scored[0], trial
    return AlignmentParams(round(best.azimuth_deg, 2), round(best.hfov_deg, 2),
                           round(best.pitch_deg, 2), round(best.roll_deg, 2))


def build_fov_grid(fov_hint_deg: float | None = None,
                   fov_margin_deg: float = 10.0,
                   ratio: float | None = None) -> np.ndarray:
    """Valores de FOV a explorar.

    Sin pista: rejilla GEOMÉTRICA de 10 a 80° con ratio constante, que da
    resolución RELATIVA constante. Con paso lineal de 2.5°, el paso vale un
    3% a 75° pero un 17% a 15° — justo donde más precisión hace falta, porque
    un teleobjetivo cambia mucho de escala con poca variación de grados.

    Con pista: rejilla lineal fina dentro del margen. Cuando el usuario sabe
    el campo aproximado interesa resolución absoluta, no relativa.
    """
    if fov_hint_deg is not None:
        lo = max(fov_hint_deg - fov_margin_deg, FOV_RANGE_DEG[0])
        hi = min(fov_hint_deg + fov_margin_deg, FOV_RANGE_DEG[1])
        step = max(0.5, (hi - lo) / 24.0)
        return np.arange(lo, hi + 1e-9, step)
    lo, hi = FOV_RANGE_DEG
    step_ratio = ratio or FOV_GRID_RATIO
    count = int(math.ceil(math.log(hi / lo) / math.log(step_ratio)))
    return np.minimum(lo * step_ratio ** np.arange(count + 1), hi)


def search_alignment(azimuths_deg: np.ndarray, elevations_deg: np.ndarray,
                     skyline_cols_px: np.ndarray, skyline_rows_px: np.ndarray,
                     width_px: int, height_px: int,
                     center_az_deg: float | None = 0.0,
                     az_margin_deg: float = COARSE_AZ_HALF_DEG,
                     fov_hint_deg: float | None = None,
                     fov_margin_deg: float = 10.0,
                     max_columns: int = 400,
                     coarse_columns: int = 200,
                     max_candidates: int = 5,
                     min_separation_deg: float = AMBIGUITY_AZ_DEG,
                     ) -> SearchResult:
    """Fuerza bruta en dos etapas sobre los cuatro parámetros.

    **center_az_deg=None significa BARRIDO COMPLETO de 360°.** Sin brújula
    EXIF y sin pista del usuario, centrar la búsqueda en un 0 que no
    significa nada y explorar ±20° alrededor es peor que inútil: el azimut
    verdadero casi nunca está ahí y el buscador devuelve con aplomo un sector
    arbitrario (medido con teleobjetivos de Sierra Nevada y de Nerja). Con
    pista (EXIF, --az-hint o sector marcado) se mantiene el margen acotado.

    Gruesa: rejilla de azimut × FOV. NI la inclinación NI el giro se
    rejillan: ambos se resuelven en forma cerrada a partir del residuo
    (ordenada = inclinación, pendiente = giro, ver solve_pitch_roll), lo que
    de paso elimina la dimensión de giro de la rejilla y paga el coste del
    barrido de 360°. Fina: rejilla de azimut × FOV alrededor de cada
    candidato, otra vez con inclinación y giro cerrados.

    Devuelve hasta `max_candidates` candidatos SEPARADOS ENTRE SÍ al menos
    `min_separation_deg` en azimut — no los mejores absolutos, que suelen ser
    variaciones del mismo óptimo. En un barrido completo la ambigüedad es la
    norma, así que lo útil es ofrecer hipótesis realmente distintas para que
    el usuario elija mirando los topónimos, que es el criterio que funciona.
    """
    cols_all = np.asarray(skyline_cols_px, dtype=float)
    rows_all = np.asarray(skyline_rows_px, dtype=float)
    full_sweep = center_az_deg is None

    def _thin(count):
        if cols_all.size <= count:
            return cols_all, rows_all
        keep = np.linspace(0, cols_all.size - 1, count).astype(int)
        return cols_all[keep], rows_all[keep]

    # la gruesa usa menos columnas: el espacio explorado creció ~3x al
    # ampliar el rango de FOV y el coste se compensa aquí, no en la fina
    coarse_cols, coarse_rows = _thin(coarse_columns)
    cols, rows = _thin(max_columns)
    min_columns = max(20, int(0.3 * cols.size))
    coarse_min_columns = max(15, int(0.3 * coarse_cols.size))

    # el perfil se diezma para la etapa gruesa: con decenas de miles de
    # combinaciones, proyectar 1800 muestras en cada una domina el coste, y
    # para ORDENAR candidatos no hace falta resolución de 0.2°. La fina usa
    # el perfil completo.
    coarse_profile_az = azimuths_deg[::COARSE_PROFILE_STRIDE]
    coarse_profile_elev = elevations_deg[::COARSE_PROFILE_STRIDE]

    # --- etapa gruesa ---
    coarse: list[tuple[float, float, AlignmentParams]] = []
    if full_sweep:
        # 360° con paso más grueso: el objetivo es localizar REGIONES
        # candidatas, y el refinado posterior afina cada una
        az_grid = np.arange(0.0, 360.0, SWEEP_AZ_STEP_DEG)
    else:
        az_grid = np.arange(center_az_deg - az_margin_deg,
                            center_az_deg + az_margin_deg + 1e-9,
                            COARSE_AZ_STEP_DEG)
    fov_grid = build_fov_grid(fov_hint_deg, fov_margin_deg,
                              ratio=SWEEP_FOV_RATIO if full_sweep else None)
    for fov in fov_grid:
        focal_px = (width_px / 2.0) / math.tan(math.radians(fov) / 2.0)
        for az in az_grid:
            base = AlignmentParams(az % 360.0, float(fov), 0.0, 0.0)
            x_px, y_px, usable = project_profile(
                coarse_profile_az, coarse_profile_elev, base,
                width_px, height_px)
            y_line = projected_y_per_column(x_px, y_px, usable, coarse_cols)
            mask = ~np.isnan(y_line)
            if mask.sum() < coarse_min_columns:
                continue
            # inclinación Y giro en forma cerrada de una sola proyección: la
            # recta ajustada al residuo da los dos (ver solve_pitch_roll).
            # Quitar el giro de la rejilla la divide por 7 y encima es más
            # preciso que un paso de 0.5°.
            fit = _fit_pitch_roll(coarse_rows[mask] - y_line[mask],
                                  coarse_cols[mask] - width_px / 2.0)
            if fit is None:
                continue
            intercept, slope, err_px = fit
            pitch = math.degrees(math.atan(intercept / focal_px))
            pitch = min(max(pitch, -PITCH_LIMIT_DEG), PITCH_LIMIT_DEG)
            roll = min(max(math.degrees(slope), -COARSE_ROLL_HALF_DEG),
                       COARSE_ROLL_HALF_DEG)
            coarse.append((err_px, float(mask.mean()),
                           AlignmentParams(az % 360.0, float(fov),
                                           round(pitch, 3), round(roll, 3))))
    az_range = ((0.0, 360.0) if full_sweep
                else (float(az_grid[0]), float(az_grid[-1])))
    fov_range = (float(fov_grid[0]), float(fov_grid[-1]))
    if not coarse:
        return SearchResult([], False, math.inf, None, False, False,
                            az_range, fov_range)
    coarse.sort(key=lambda item: item[0])

    # Candidatos SEPARADOS ENTRE SÍ, no los mejores absolutos: en un barrido
    # completo los tres mejores suelen ser el mismo óptimo con variaciones de
    # décimas, y eso no le da a elegir nada al usuario.
    top: list[AlignmentParams] = []
    for _err_px, _coverage, params in coarse:
        if all(_az_separation(params, other) >= min_separation_deg
               for other in top):
            top.append(params)
        if len(top) == max_candidates:
            break

    # SATURACIÓN: si el óptimo se apoya en un extremo de lo explorado, el
    # verdadero puede estar FUERA y la búsqueda no tiene forma de saberlo.
    # Es lo que pasó con un teleobjetivo real (FOV ~15-20°) contra un rango
    # que empezaba en 40: devolvió 72.5, el borde opuesto.
    coarse_best = coarse[0][2]
    fov_at_edge = bool(coarse_best.hfov_deg <= fov_grid[0] + 1e-6
                       or coarse_best.hfov_deg >= fov_grid[-1] - 1e-6)
    if full_sweep:
        az_at_edge = False   # no hay borde: se exploraron los 360°
    else:
        az_offset = abs((coarse_best.azimuth_deg - center_az_deg + 180.0)
                        % 360.0 - 180.0)
        az_at_edge = bool(az_offset >= az_margin_deg - COARSE_AZ_STEP_DEG)

    # AMBIGÜEDAD: el mejor óptimo LEJANO en azimut es una hipótesis
    # genuinamente distinta, no un vecino de rejilla. Con FOV estrecho la
    # firma del horizonte es pobre y la correlación tiene muchos máximos casi
    # equivalentes; si ese óptimo lejano no es claramente peor, la búsqueda
    # no distingue entre ambos y no debe presentarse como solución.
    alternative_params = next(
        (params for _e, _c, params in coarse
         if _az_separation(params, coarse_best) > AMBIGUITY_AZ_DEG), None)

    # --- etapa fina: TODOS los candidatos, no solo el mejor ---
    # el FOV se acota al rango REALMENTE explorado: sin esto la etapa fina se
    # escapaba por encima del máximo (medido: 82.5 con tope 80), reportando
    # un valor fuera del espacio que la propia salida dice haber explorado y
    # desactivando de paso la detección de saturación
    fov_lo, fov_hi = float(fov_grid[0]), float(fov_grid[-1])
    az_step = SWEEP_AZ_STEP_DEG if full_sweep else COARSE_AZ_STEP_DEG
    fov_span = max(fov_hi - fov_lo, 1.0) * 0.08
    top = [_refine(azimuths_deg, elevations_deg, params, cols, rows,
                   width_px, height_px, min_columns, az_step, fov_span,
                   fov_lo, fov_hi)
           for params in top]

    # Todos se puntúan con la MISMA métrica exacta antes de ordenar: mezclar
    # un refinado con estimaciones de la rejilla gruesa (que ya descontaban
    # el desplazamiento óptimo) daba una lista cuyo orden no se correspondía
    # con los errores mostrados.
    scored_top: list[SearchCandidate] = []
    for params in top:
        scored = _score_exact(azimuths_deg, elevations_deg, params, cols, rows,
                              width_px, height_px, min_columns)
        if scored is None:
            continue
        scored_top.append(SearchCandidate(
            params, scored[0], _px_to_deg(scored[0], params, width_px),
            scored[1], scored[2]))
    scored_top.sort(key=lambda candidate: candidate.error_px)

    # la alternativa lejana se puntúa con la MISMA métrica exacta que el
    # mejor: comparar una estimación gruesa con un refinado inventaría un
    # margen que no existe
    alternative = None
    margin = math.inf
    if alternative_params is not None and scored_top:
        # la alternativa se REFINA igual que el mejor antes de puntuarla:
        # comparar un refinado contra una estimación de rejilla infla el
        # margen y hace pasar por inequívoco lo que no lo es (medido: 0.43
        # sin refinar frente a un caso genuinamente ambiguo)
        alternative_params = _refine(
            azimuths_deg, elevations_deg, alternative_params, cols, rows,
            width_px, height_px, min_columns, az_step, fov_span,
            fov_lo, fov_hi)
        scored = _score_exact(azimuths_deg, elevations_deg, alternative_params,
                              cols, rows, width_px, height_px, min_columns)
        if scored is not None:
            alternative = SearchCandidate(
                alternative_params, scored[0],
                _px_to_deg(scored[0], alternative_params, width_px),
                scored[1], scored[2])
            best_px = scored_top[0].error_px
            margin = ((alternative.error_px - best_px) / best_px
                      if best_px > 0 else math.inf)
    return SearchResult(candidates=scored_top,
                        ambiguous=bool(margin < AMBIGUITY_MARGIN),
                        ambiguity_margin=float(margin),
                        alternative=alternative,
                        fov_at_edge=fov_at_edge, az_at_edge=az_at_edge,
                        az_range_deg=az_range, fov_range_deg=fov_range)


def _az_separation(a: AlignmentParams, b: AlignmentParams) -> float:
    return abs((a.azimuth_deg - b.azimuth_deg + 180.0) % 360.0 - 180.0)


def solve_pitch_roll(azimuths_deg: np.ndarray, elevations_deg: np.ndarray,
                     params: AlignmentParams, skyline_cols_px: np.ndarray,
                     skyline_rows_px: np.ndarray, width_px: int,
                     height_px: int, iterations: int = 2,
                     min_columns: int = 20) -> tuple[float, float] | None:
    """Inclinación y giro en forma cerrada, fijados azimut y FOV.

    Modelo de primer orden, verificado numéricamente contra project_profile:

        y(pitch, roll) − y(0, 0)  ≈  f·tan(pitch) + (x − W/2)·roll_rad

    o sea que el residuo vertical entre la cresta detectada y la línea
    proyectada es una RECTA en x: su ordenada da la inclinación y su
    pendiente el giro. Dos parámetros que se resuelven de una vez en lugar de
    buscarse a mano.

    Se itera (defecto 2) porque el modelo es de primer orden y a 8°/5° deja
    ~0.1° de error; con una segunda pasada el residuo del modelo desaparece.
    Las columnas cuyo residuo se aparta más de 3·MAD de la mediana se
    descartan: un arbusto o un tejado detectado como cresta no debe torcer la
    recta.

    Devuelve (pitch_deg, roll_deg) o None si no quedan bastantes columnas
    utilizables — mejor no resolver que resolver con cuatro puntos.
    """
    cols = np.asarray(skyline_cols_px, dtype=float)
    rows = np.asarray(skyline_rows_px, dtype=float)
    focal_px = (width_px / 2.0) / math.tan(math.radians(params.hfov_deg) / 2.0)
    pitch_deg, roll_deg = params.pitch_deg, params.roll_deg
    used = 0

    for _ in range(max(1, iterations)):
        current = AlignmentParams(params.azimuth_deg, params.hfov_deg,
                                  pitch_deg, roll_deg)
        x_px, y_px, usable = project_profile(azimuths_deg, elevations_deg,
                                             current, width_px, height_px)
        y_line = projected_y_per_column(x_px, y_px, usable, cols)
        mask = ~np.isnan(y_line)
        if mask.sum() < min_columns:
            return None
        residual = rows[mask] - y_line[mask]
        offset = cols[mask] - width_px / 2.0

        # rechazo robusto: la mediana y su MAD no las mueve un puñado de
        # columnas disparatadas, la recta de mínimos cuadrados sí
        deviation = np.abs(residual - np.median(residual))
        scale = 1.4826 * np.median(deviation)
        keep = deviation <= 3.0 * scale if scale > 0 else np.ones_like(
            residual, dtype=bool)
        if keep.sum() < min_columns:
            keep = np.ones_like(residual, dtype=bool)
        used = int(keep.sum())

        design = np.vstack([np.ones(used), offset[keep]]).T
        (intercept, slope), *_ = np.linalg.lstsq(design, residual[keep],
                                                 rcond=None)
        pitch_deg += math.degrees(math.atan(intercept / focal_px))
        roll_deg += math.degrees(slope)
        # acotado a lo que una cámara puede hacer: sin esto la búsqueda
        # devolvía giros de −30°, un grado de libertad falso con el que el
        # ajuste se retuerce hasta encajar ruido
        pitch_deg = min(max(pitch_deg, -PITCH_LIMIT_DEG), PITCH_LIMIT_DEG)
        roll_deg = min(max(roll_deg, -ROLL_LIMIT_DEG), ROLL_LIMIT_DEG)

    if used < min_columns:
        return None
    return pitch_deg, roll_deg


def sector_to_ranges(start_deg: float, end_deg: float,
                     ) -> tuple[float, float, float, float, float, float]:
    """Arco marcado por el usuario -> rangos de los sliders.

    Devuelve (az_centro, az_ancho, az_lo, az_hi, fov_lo, fov_hi).

    Un arco de MÁS DE 180° se interpreta como el COMPLEMENTARIO, es decir el
    que cruza el norte: arrastrar de 10° a 350° selecciona los 20° de
    enfrente, no los 340° del medio. La interpretación es inequívoca SOLO
    porque FOV_RANGE_DEG tope en 80°: ninguna foto abarca 340°, así que la
    selección grande no puede ser literal. Si algún día se subiera ese tope
    por encima de 180°, esta regla dejaría de valer.
    """
    start = start_deg % 360.0
    end = end_deg % 360.0
    lo, hi = min(start, end), max(start, end)
    width = hi - lo
    if width > 180.0:
        lo, hi = hi, lo + 360.0     # el complementario, cruzando el norte
        width = hi - lo
    width = max(width, 1.0)
    center = (lo + width / 2.0) % 360.0
    fov_lo = max(width * 0.5, FOV_RANGE_DEG[0])
    fov_hi = min(max(width * 2.0, fov_lo + 1.0), FOV_RANGE_DEG[1])
    return center, width, lo, hi, fov_lo, fov_hi


def _score_exact(azimuths_deg, elevations_deg, params, cols, rows,
                 width_px, height_px, min_columns):
    """(error px, cobertura, fracción saturada) — px para comparar."""
    x_px, y_px, usable = project_profile(azimuths_deg, elevations_deg, params,
                                         width_px, height_px)
    y_line = projected_y_per_column(x_px, y_px, usable, cols)
    mask = ~np.isnan(y_line)
    if mask.sum() < min_columns:
        return None
    residual = rows[mask] - y_line[mask]
    return (_clipped_mean_abs(residual), float(mask.mean()),
            _saturated_fraction(residual))


def _px_to_deg(err_px: float, params: AlignmentParams, width_px: int) -> float:
    """Solo para MOSTRAR el error en grados; la comparación entre candidatos
    es siempre en píxeles."""
    focal_px = (width_px / 2.0) / math.tan(math.radians(params.hfov_deg) / 2.0)
    return math.degrees(math.atan(err_px / focal_px))


CLIP_PX = 40.0        # tope del residuo por columna
SATURATED_FRACTION = 0.5   # por encima, el error deja de medir la calidad


def _clipped_mean_abs(residual_px: np.ndarray, clip_px: float = CLIP_PX) -> float:
    """Media del residuo absoluto, recortado.

    Media y no mediana: sobre filas cuantizadas por el submuestreo del
    detector, la mediana forma mesetas y los empates se resuelven por orden
    de iteración — la búsqueda quedaba sesgada un paso de rejilla. La media
    es sensible a subpíxel; el recorte acota las columnas donde el detector
    se fue a un tejado o un árbol, que es de lo que protegía la mediana.

    CUIDADO al leer el número: el recorte satura. Un ajuste con residuos de
    350 px da ~40 igual que uno con residuos de 45, así que un error cercano
    a CLIP_PX NO significa "casi bueno" sino "sin medir". Por eso
    SearchCandidate lleva `saturated_fraction`: es lo que distingue un
    ajuste mediocre de uno catastrófico.
    """
    return float(np.mean(np.minimum(np.abs(residual_px), clip_px)))


def _saturated_fraction(residual_px: np.ndarray,
                        clip_px: float = CLIP_PX) -> float:
    """Fracción de columnas cuyo residuo toca el tope del recorte."""
    if residual_px.size == 0:
        return 1.0
    return float(np.mean(np.abs(residual_px) >= clip_px))


def _is_distinct(a: AlignmentParams, b: AlignmentParams) -> bool:
    delta_az = abs((a.azimuth_deg - b.azimuth_deg + 180.0) % 360.0 - 180.0)
    return delta_az > 1.5 or abs(a.hfov_deg - b.hfov_deg) > 5.0


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(np.asarray(values, dtype=float), pad, mode="edge")
    return np.median(sliding_window_view(padded, window), axis=-1)
