"""Proyección del perfil de horizonte sobre una fotografía y persistencia
del alineamiento manual (fase 2, previa al ML).

Convenios (sufijos obligatorios, ver CLAUDE.md):
- azimuth_deg: azimut del CENTRO de la imagen, no del borde.
- pitch_deg: elevación del centro de la imagen; positivo = cámara mirando
  hacia arriba.
- roll_deg: positivo = el lado derecho del horizonte dibujado BAJA.
- Proyección pinhole rectilínea, sin distorsión de lente (documentado: ese
  residuo lo absorberá el alineamiento automático de la fase 2).

El JSON de alineamiento guarda, además de los cuatro parámetros finales, un
bloque `seed` con los valores iniciales y su procedencia ("exif", "default",
"cli"): la diferencia entre el azimut ajustado y el azimut EXIF es la medida
del error de la brújula, que es el dato que la fase 2 necesita. `notes` y
`created_utc` anotan las condiciones de cada captura: este es el conjunto de
validación del ML y conviene que nazca anotado.
"""

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ALIGN_VERSION = 1

# tags EXIF (los sub-IFD donde viven varían entre cámaras: _exif_lookup
# busca en el IFD base y en el sub-IFD Exif)
_IFD_EXIF = 0x8769
_IFD_GPS = 0x8825
_TAG_FOCAL_35MM = 41989
_TAG_GPS_LAT_REF, _TAG_GPS_LAT = 1, 2
_TAG_GPS_LON_REF, _TAG_GPS_LON = 3, 4
_TAG_GPS_DIR = 17

_FULL_FRAME_WIDTH_MM = 36.0


@dataclass
class AlignmentParams:
    azimuth_deg: float
    hfov_deg: float
    pitch_deg: float
    roll_deg: float


def project_profile(azimuths_deg: np.ndarray, elevations_deg: np.ndarray,
                    params: AlignmentParams, width_px: int, height_px: int,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Proyecta muestras (azimut, elevación) a píxeles de la foto.

    Devuelve (x_px, y_px, delante): `delante` marca las muestras por delante
    de la cámara; las que no, y las de elevación NaN, salen False y sus
    píxeles no son utilizables.

    Pinhole completa, no aproximación lineal: con FOV de 65° el error de la
    lineal en los bordes es ~4% del ancho (~150 px en una foto de 4000), que
    arruinaría el alineamiento fino.
    """
    el_rad = np.radians(np.asarray(elevations_deg, dtype=float))
    daz_rad = np.radians(
        (np.asarray(azimuths_deg, dtype=float) - params.azimuth_deg + 180.0)
        % 360.0 - 180.0)

    forward = np.cos(el_rad) * np.cos(daz_rad)
    right = np.cos(el_rad) * np.sin(daz_rad)
    up = np.sin(el_rad)

    # inclinación: rotación en el plano adelante/arriba
    pitch_rad = math.radians(params.pitch_deg)
    forward_p = forward * math.cos(pitch_rad) + up * math.sin(pitch_rad)
    up_p = -forward * math.sin(pitch_rad) + up * math.cos(pitch_rad)

    # giro: rotación en el plano derecha/arriba; con roll positivo un punto
    # a la derecha del centro obtiene up_cam negativo -> se dibuja más bajo
    roll_rad = math.radians(params.roll_deg)
    right_r = right * math.cos(roll_rad) + up_p * math.sin(roll_rad)
    up_r = -right * math.sin(roll_rad) + up_p * math.cos(roll_rad)

    focal_px = (width_px / 2.0) / math.tan(math.radians(params.hfov_deg) / 2.0)
    delante = forward_p > 1e-9  # NaN comparado da False: los voids caen aquí
    with np.errstate(divide="ignore", invalid="ignore"):
        x_px = width_px / 2.0 + focal_px * right_r / forward_p
        y_px = height_px / 2.0 - focal_px * up_r / forward_p
    return x_px, y_px, delante


def projected_y_per_column(x_px: np.ndarray, y_px: np.ndarray,
                           usable: np.ndarray,
                           columns_px: np.ndarray) -> np.ndarray:
    """Altura de la silueta en cada columna pedida, interpolando la
    polilínea proyectada. NaN fuera del tramo que cubre.

    La proyección da un punto por muestra de azimut, no por columna; la tira
    de recorte y la detección de borde necesitan lo segundo.
    """
    xs = np.asarray(x_px)[usable]
    ys = np.asarray(y_px)[usable]
    columns_px = np.asarray(columns_px, dtype=float)
    if xs.size < 2:
        return np.full(columns_px.shape, np.nan)
    order = np.argsort(xs)
    return np.interp(columns_px, xs[order], ys[order],
                     left=np.nan, right=np.nan)


def load_oriented_photo(photo_path: str) -> tuple[Image.Image, np.ndarray]:
    """ÚNICO punto de carga de píxeles de la foto: aplica la orientación EXIF
    una sola vez y devuelve (imagen, array) del mismo origen.

    Las cámaras de móvil no rotan los píxeles: guardan la rotación en el tag
    Orientation. Todo consumidor de píxeles (canvas principal, tira, volcados
    de depuración) debe salir de aquí — una segunda ruta de carga es una
    oportunidad de mezclar ejes orientados con píxeles sin orientar, el tipo
    de fallo plausible-pero-incorrecto contra el que está escrito CLAUDE.md.
    """
    with Image.open(photo_path) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
    array = np.asarray(img)
    # coherencia imagen<->array: si esto falla, hay dos orígenes de píxeles
    assert array.shape[:2] == (img.height, img.width), (
        f"array {array.shape[:2]} vs imagen {(img.height, img.width)}")
    return img, array


def build_strip(photo_np: np.ndarray, x_px: np.ndarray, y_px: np.ndarray,
                usable: np.ndarray, columns_px: np.ndarray, band_px: float,
                strip_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Parche de la tira de horizonte: la banda de ±band_px alrededor de la
    silueta proyectada, ENDEREZADA (la línea queda recta en el centro del
    parche) y remuestreada a strip_h filas.

    Función pura compartida por la GUI y por los volcados de verificación:
    lo que se comprueba fuera de la ventana ES lo que la ventana dibuja, no
    una réplica. Devuelve (parche RGB strip_h × len(columns_px), columnas
    con silueta proyectable). Las columnas sin silueta o fuera de la foto se
    rellenan de gris oscuro.
    """
    photo_np = np.asarray(photo_np)
    height, width = photo_np.shape[0], photo_np.shape[1]
    columns_px = np.asarray(columns_px, dtype=float)

    y_line = projected_y_per_column(x_px, y_px, usable, columns_px)
    columns_valid = ~np.isnan(y_line)

    offsets = np.linspace(-band_px, band_px, strip_h)
    rows = np.rint(np.nan_to_num(y_line)[None, :] + offsets[:, None])
    cols = np.rint(columns_px)[None, :].repeat(strip_h, axis=0)
    inside = ((rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
              & columns_valid[None, :])
    # Lo que cae FUERA de la foto se marca con TRAMA diagonal, no con un gris
    # casi negro: un relleno liso se lee como contenido, y cuando la línea
    # proyectada queda por encima del borde superior de la foto (medido:
    # y_line = −145 con campo estrecho) la tira aparentaba estar del revés,
    # con una banda oscura arriba y la montaña debajo. La trama dice "aquí no
    # hay foto", que es la verdad. Mismo convenio que el "sin dato" de render/.
    row_idx = np.arange(strip_h)[:, None]
    col_idx = np.arange(columns_px.size)[None, :]
    hatch = np.where(((row_idx + col_idx) % 14) < 4, 96, 58)
    patch = np.where(
        inside[..., None],
        photo_np[np.clip(rows, 0, height - 1).astype(int),
                 np.clip(cols, 0, width - 1).astype(int)],
        hatch[..., None]).astype(np.uint8)
    return patch, columns_valid


def load_session(photo_path: str) -> dict | None:
    """El .align.json de la foto si existe y es utilizable; None si no.

    Reanudar desde la sesión anterior en vez de las semillas: perder el
    ajuste al reabrir es inaceptable cuando la herramienta se usa sobre
    decenas de fotos.
    """
    path = alignment_json_path(photo_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != ALIGN_VERSION or "alignment" not in data:
        return None
    return data


def alignment_json_path(photo_path: str) -> Path:
    """foto.jpg -> foto.align.json, en el mismo directorio."""
    p = Path(photo_path)
    return p.with_name(p.stem + ".align.json")


def seed_entry(value: float, source: str) -> dict:
    """Un valor inicial con su procedencia: "exif", "default" o "cli"."""
    return {"value": float(value), "source": source}


def provenance_entry(search_used: bool = False, auto_pitch_roll: bool = False,
                     detector: str | None = None,
                     manual_review: bool = False,
                     review_delta: dict | None = None,
                     review_note: str | None = None,
                     review_kind: str | None = None) -> dict:
    """Cómo se obtuvo el alineamiento. Decide si sirve como VERDAD.

    LO QUE CONTAMINA NO ES USAR LA BÚSQUEDA, SINO ACEPTARLA SIN REVISAR.
    Esta distinción costó una vuelta atrás: el primer criterio descartaba
    toda referencia que hubiera pasado por el buscador, y eso tira a la
    basura trabajo humano legítimo. Verificar dónde caen los TOPÓNIMOS es
    información independiente del detector —de hecho es el criterio que
    CLAUDE.md señala como el que de verdad zanja entre hipótesis—, así que
    un alineamiento revisado así no hereda el sesgo del detector aunque el
    punto de partida saliera de él. Casos reales: en `sierra` el FOV se
    corrigió de 78.5 a 74.7 a mano al ver que las etiquetas no caían sobre
    las cimas; en `141720` el azimut se zanjó comparando topónimos entre tres
    candidatos, no por la métrica de píxeles.

    Lo que sí queda inservible como verdad es aceptar la salida del buscador
    tal cual: ahí el detector se estaría midiendo contra su propia respuesta.

    - `manual`: no intervino ninguna ayuda automática.
    - `manual_review`: hubo ayuda, pero el usuario la revisó después.
      `review_delta` guarda cuánto movió cada parámetro, para que quien lea
      el informe juzgue.
    - `review_kind`: `"correccion"` si la revisión cambió el resultado,
      `"verificacion"` si lo dio por bueno sin tocarlo, `None` si no consta.
      NO se deduce de que `review_delta` esté vacío: en `141720` la revisión
      corrigió de verdad —cambió de hipótesis entre tres candidatos por los
      topónimos— y aun así no hay delta numérico, porque no fue mover un
      valor. Confundir ambas cosas hace afirmaciones falsas sobre el sesgo.
    - `reviewed`: la conclusión, y lo que mira el arnés.
    """
    return {
        "search_used": bool(search_used),
        "auto_pitch_roll": bool(auto_pitch_roll),
        "detector": detector,
        "manual": not (search_used or auto_pitch_roll),
        "manual_review": bool(manual_review),
        "review_delta": review_delta or {},
        "review_note": review_note,
        "review_kind": review_kind,
        # sirve como verdad si nadie automático la tocó, o si un humano la
        # revisó después con información ajena al detector
        "reviewed": (not (search_used or auto_pitch_roll)
                     or bool(manual_review)),
    }


_NOTE_MARKS = ("busqueda automatica", "búsqueda automática", "buscador")


def infer_provenance(data: dict, tolerance: float = 0.051) -> dict:
    """Procedencia de un alineamiento. Deducida si el JSON no trae el campo
    `provenance`, y en ese caso NUNCA cuenta como revisada.

    La deducción compara `alignment` con `seed`, lo que detecta el caso en que
    el bloque `seed` ES la salida del buscador, pero se le escapan variantes
    reales: una semilla del buscador retocada después a mano ya no coincide,
    y una búsqueda lanzada desde la GUI deja en `seed` los valores previos.
    Por eso solo informa (`hint`) y no decide.

    Que no decida NO significa que la referencia se descarte: significa que
    hay que anotar a mano cómo se verificó. Es lo que se hizo con las cinco
    primeras, todas revisadas contra topónimos.
    """
    if isinstance(data.get("provenance"), dict):
        known = dict(data["provenance"])
        # compatibilidad con los provenance escritos antes de `reviewed`
        known.setdefault("manual_review", False)
        known.setdefault("reviewed",
                         known.get("manual", False)
                         or known.get("manual_review", False))
        return {**known, "inferred": False, "unknown": False}

    alignment = data.get("alignment") or {}
    seed = data.get("seed") or {}
    comparables = []
    for name in ("azimuth_deg", "hfov_deg", "pitch_deg", "roll_deg"):
        entry, final = seed.get(name), alignment.get(name)
        if not isinstance(entry, dict) or final is None:
            comparables = []
            break
        if entry.get("source") == "default" and entry.get("value") == 0.0:
            continue          # un 0 por defecto no coincide, coincide por azar
        comparables.append(abs(float(entry["value"]) - float(final))
                           <= tolerance)
    es_seed = bool(comparables) and all(comparables)
    notas = (data.get("notes") or "").lower()
    por_notas = any(marca in notas for marca in _NOTE_MARKS)

    if es_seed:
        hint = "el alignment ES el seed del buscador"
    elif por_notas:
        hint = "las notas mencionan la búsqueda automática"
    else:
        hint = "sin indicios, pero tampoco constancia de que sea manual"
    return {"search_used": es_seed or por_notas,
            "auto_pitch_roll": es_seed,
            "detector": None, "manual": False, "manual_review": False,
            "review_delta": {}, "review_note": None, "review_kind": None,
            "reviewed": False,
            "inferred": True, "unknown": True, "hint": hint}


def save_alignment(json_path: str, photo_name: str, observer: dict,
                   seed: dict, params: AlignmentParams, width_px: int,
                   height_px: int, notes: str = "",
                   created_utc: str | None = None,
                   low_confidence: bool = False,
                   provenance: dict | None = None) -> None:
    """Escribe el JSON de alineamiento. `photo_name` debe ser relativo al
    directorio del JSON, para que el par foto+alineamiento se mueva junto."""
    payload = {
        "version": ALIGN_VERSION,
        "photo": photo_name,
        "created_utc": created_utc or _utc_now(),
        "notes": notes,
        # Procedencia: manual, o tocado por búsqueda/ajuste automático. Sin
        # esto no se puede saber si una referencia sirve como verdad para
        # evaluar detectores (ver provenance_entry).
        "provenance": provenance or provenance_entry(),
        # Referencia dudosa: el alineamiento existe pero no es fiable como
        # VERDAD para evaluar detectores (p. ej. hecho a mano porque el
        # detector automático fallaba en esa misma foto). Medir contra una
        # referencia mala es peor que no medir: un detector mejor mediría
        # peor. El arnés la reporta aparte y la excluye del veredicto.
        "low_confidence": bool(low_confidence),
        "observer": observer,
        "seed": seed,
        "alignment": asdict(params),
        # exif_oriented: las dimensiones (y los parámetros de alineamiento)
        # corresponden a la imagen CON la orientación EXIF ya aplicada. Quien
        # relea la foto debe aplicar ImageOps.exif_transpose o equivalente;
        # los lectores que entregan el buffer crudo (cv2.imread, por ejemplo)
        # darían ancho y alto intercambiados en fotos verticales y todo el
        # alineamiento saldría plausible pero incorrecto.
        "image": {"width_px": width_px, "height_px": height_px,
                  "exif_oriented": True},
    }
    Path(json_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def load_alignment(json_path: str) -> dict:
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def seed_from_exif(photo_path: str) -> dict:
    """Semillas disponibles en el EXIF de la foto: cualquier subconjunto de
    azimuth_deg (GPSImgDirection), hfov_deg (focal equivalente 35 mm),
    lat_deg y lon_deg (GPS). Ausencias no son error: dict sin esa clave."""
    from PIL import Image
    with Image.open(photo_path) as img:
        return _seed_from_exif(img.getexif())


def _seed_from_exif(exif) -> dict:
    """Semillas del EXIF. Un valor inválido NUNCA sale como número: o sale
    bien, o no sale la clave y se marca `gps_invalid`.

    Medido en fotos reales: hay cámaras que escriben el bloque GPS con
    racionales 0/0 y el hemisferio a '\\x00' cuando no llegaron a fijar
    posición. Eso producía lat/lon = NaN, que entraban en el pipeline sin
    que nada fallara — el fallo callado que CLAUDE.md prohíbe.
    """
    out: dict = {}
    focal_35 = _exif_lookup(exif, _TAG_FOCAL_35MM)
    if focal_35:
        try:
            focal = float(focal_35)
        except (TypeError, ValueError, ZeroDivisionError):
            focal = 0.0
        if math.isfinite(focal) and focal > 0.0:
            out["hfov_deg"] = math.degrees(
                2.0 * math.atan(_FULL_FRAME_WIDTH_MM / (2.0 * focal)))
    try:
        gps = exif.get_ifd(_IFD_GPS)
    except (AttributeError, KeyError):
        gps = {}

    direction = _safe_float(gps.get(_TAG_GPS_DIR))
    if direction is not None:
        out["azimuth_deg"] = direction % 360.0

    lat = _dms_to_deg(gps.get(_TAG_GPS_LAT), gps.get(_TAG_GPS_LAT_REF))
    lon = _dms_to_deg(gps.get(_TAG_GPS_LON), gps.get(_TAG_GPS_LON_REF))
    if lat is not None and lon is not None and abs(lat) <= 90.0 \
            and abs(lon) <= 180.0:
        out["lat_deg"], out["lon_deg"] = lat, lon
    elif _TAG_GPS_LAT in gps or _TAG_GPS_LON in gps:
        # el bloque existe pero no es utilizable: se dice, no se calla
        out["gps_invalid"] = True
    return out


def _safe_float(value) -> float | None:
    """float() que devuelve None en vez de NaN, infinito o excepción.

    Los racionales EXIF pueden venir con denominador 0; pillow los convierte
    en NaN y a partir de ahí el número contamina todo sin fallar."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) else None


def _exif_lookup(exif, tag: int):
    """Busca un tag en el IFD base y en el sub-IFD Exif: según la cámara
    (y según si pillow lo escribió plano) puede estar en cualquiera."""
    value = exif.get(tag)
    if value is not None:
        return value
    try:
        return exif.get_ifd(_IFD_EXIF).get(tag)
    except (AttributeError, KeyError):
        return None


def _dms_to_deg(dms, ref) -> float | None:
    """Grados,minutos,segundos -> grados decimales, o None si no es usable.

    El hemisferio tiene que ser N/S/E/W explícito: hay cámaras que dejan un
    '\\x00' cuando no fijaron posición, y tomarlo por "norte/este" daría una
    coordenada plausible pero inventada."""
    if dms is None or ref is None:
        return None
    try:
        parts = [_safe_float(v) for v in dms]
    except TypeError:
        return None
    if len(parts) != 3 or any(p is None for p in parts):
        return None
    hemisphere = str(ref).strip().upper()
    if hemisphere not in ("N", "S", "E", "W"):
        return None
    value = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    return -value if hemisphere in ("S", "W") else value


def _utc_now() -> str:
    return (datetime.now(timezone.utc).isoformat(timespec="seconds")
            .replace("+00:00", "Z"))
