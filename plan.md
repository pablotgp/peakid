# Plan: src/geo/ — cuatro funciones públicas de geometría

## Contexto

Primer módulo real del proyecto PeakID. `geo/` no depende de nadie y todo lo demás
dependerá de él, así que un error de convenio aquí contamina todo. CLAUDE.md fija
los convenios (grados en API pública, azimut horario desde el norte, curvatura+
refracción siempre, sufijos de unidad obligatorios) y los tests dorados que actúan
de contrato. Este plan implementa las cuatro funciones y sus tests dorados.

## Fichero

Todo en [src/geo/__init__.py](src/geo/__init__.py) (hoy vacío). Son fórmulas de
pocas líneas; no hace falta submódulo.

Constantes de módulo:

```python
R_EARTH_M = 6_371_000.0   # radio terrestre medio
K_REFRACTION = 0.13       # coeficiente de refracción estándar
```

## Firmas

```python
def haversine_m(lat_a_deg: float, lon_a_deg: float,
                lat_b_deg: float, lon_b_deg: float) -> float
    # distancia en metros sobre R_EARTH_M

def azimuth_deg(lat_a_deg: float, lon_a_deg: float,
                lat_b_deg: float, lon_b_deg: float) -> float
    # azimut inicial A→B, [0, 360), horario desde el norte

def curvature_drop_m(d_m: float) -> float
    # (1 - K_REFRACTION) * d_m**2 / (2 * R_EARTH_M)

def elevation_deg(d_m: float, h_a_m: float, h_b_m: float) -> float
    # degrees(atan2(h_b_m - h_a_m - curvature_drop_m(d_m), d_m))
```

Decisiones de firma:

- Coordenadas como cuatro floats en orden `(lat, lon)` de A y luego de B —
  nunca tuplas `(lon, lat)`.
- `elevation_deg` recibe la **distancia**, no las coordenadas: el barrido de
  horizonte la llamará millones de veces con `d_m` ya conocido (paso de 30 m a
  lo largo del rayo). Quien tenga solo coordenadas compone
  `elevation_deg(haversine_m(...), h_a, h_b)`.
- `curvature_drop_m` sin parámetro `k`: la refracción va SIEMPRE incluida
  (convenio no negociable); `K_REFRACTION` es constante de módulo, no argumento.
- No existe variante pública "sin curvatura". El valor de control 1.93° del test
  dorado 2 se calcula inline en el test con `atan2(Δh, d)`.

## Convenios de ángulo por paso

1. **Entrada** en grados decimales (N+, E+). Primera línea de cada función:
   `math.radians(...)` a variables con sufijo `_rad`.
2. **Cuerpo** íntegro en radianes.
3. **Salida**: `math.degrees(...)`. Solo el azimut se normaliza con `% 360.0`
   (atan2 devuelve (−180, 180]; en Python el módulo de un negativo es positivo,
   así −24° → 336°, que es lo que espera el test dorado de Peñalara).
   La elevación NO se normaliza: vive en (−90, 90) y puede ser negativa;
   un `% 360` la rompería.

## Orden de argumentos en atan2 (Python: `atan2(y, x)`)

- **Azimut**: `atan2(sin(Δλ)·cos(φ_b),  cos(φ_a)·sin(φ_b) − sin(φ_a)·cos(φ_b)·cos(Δλ))`
  El primer argumento (y) es la componente **Este** del rumbo; el segundo (x) es
  la componente **Norte**. `atan2(E, N)` mide el ángulo desde el norte abriéndose
  hacia el este = horario = azimut geográfico. Invertirlos daría el convenio
  matemático (antihorario desde el Este) y fallaría el test de sentido.
- **Elevación**: `atan2(h_b − h_a − drop_m,  d_m)` — vertical primero (cateto
  opuesto), distancia horizontal segundo (cateto adyacente). `d_m > 0` siempre,
  así que el resultado ya cae en (−90°, 90°).

## Tests — tests/test_golden.py (nuevo)

Casos 1–4 del contrato de CLAUDE.md (los que no requieren DEM ni visibilidad):

1. Curvatura: 10 km → 6.8 m, 30 km → 61.5 m, 60 km → 245.9 m (±2%).
2. Puerta del Sol (40.4168, −3.7038, 650 m) → Peñalara (40.8508, −3.9578, 2428 m):
   distancia ≈ 52.9 km ±1 km; azimut ≈ 336° ±1°; elevación con curvatura
   ≈ 1.72° ±0.05°; control sin curvatura ≈ 1.93° (calculado inline en el test).
3. Simetría: `azimuth_deg(A→B)` y `azimuth_deg(B→A)` difieren 180° ±0.5°.
4. Sentido: azimut hacia un punto justo al norte ≈ 0°; hacia un punto justo al
   este ≈ 90° (verifica el orden de atan2 sin necesitar función de destino).

## Verificación

```
python -m pytest tests/test_golden.py -v
```

El módulo no se da por terminado hasta que estos tests pasen. Sin dependencias
externas (solo `math` de stdlib; `pytest` para tests).

---

# Parte 2: src/dem/ — lectura e interpolación de tiles .hgt

## Contexto

`data/` tiene 23 tiles SRTM1 (N36–N39, W1–W6), confirmado `N38W004.hgt` =
25 934 402 bytes exactos. **Falta `N40W004.hgt`**, que es el tile que cubre
Peñalara (40.8508, −3.9578) — el test dorado 5 (altitud DEM) va a lanzar
`TileNotFoundError` hasta que se descargue ese tile. No es un bug, es cobertura
pendiente.

## Fichero

[src/dem/__init__.py](src/dem/__init__.py) (hoy vacío).

## Firmas

```python
class TileNotFoundError(FileNotFoundError):
    """El .hgt esperado para unas coordenadas no está en dem_dir."""

@dataclass
class Tile:
    array: np.memmap   # shape (n+1, n+1), dtype '>i2', big-endian sin copiar
    lat_sw: int
    lon_sw: int
    n: int              # tamaño - 1 (3600 en SRTM1, 1200 en SRTM3)

def load_tile(path: str) -> Tile
    # deduce SRTM1 (25_934_402 B, n=3600) vs SRTM3 (2_884_802 B, n=1200)
    # por os.path.getsize(path); cualquier otro tamaño -> ValueError

def locate_tile_path(lat_deg: float, lon_deg: float, dem_dir: str = "data") -> str
    # nombre esperado a partir de floor(lat), floor(lon) (esquina SO)

def get_tile(lat_deg: float, lon_deg: float, dem_dir: str = "data") -> Tile
    # caché de módulo _tile_cache: dict[str, Tile], clave = nombre de fichero
    # si el fichero no existe -> TileNotFoundError con ruta esperada y coords

def elevation_m(lat_deg: float, lon_deg: float, dem_dir: str = "data") -> float | None
    # bilineal; None si todas las esquinas usadas son void
```

## Orden de bytes

`dtype='>i2'` (int16 big-endian) directamente en `np.memmap` — numpy interpreta
los bytes al indexar, sin byteswap manual. `memmap` en vez de cargar el array
entero: solo se traen a RAM las páginas consultadas (tiles de ~26 MB, un barrido
de horizonte toca muchos).

Verificación de tamaño exacto al abrir (`os.path.getsize`): 25 934 402 B → SRTM1
(n=3600), 2 884 802 B → SRTM3 (n=1200), cualquier otro tamaño → `ValueError`.

## Conversión (lat, lon) → fila/columna

Fórmula celda→coordenada de CLAUDE.md (`n = size − 1`):

    lat = lat_sw + 1 − row/n
    lon = lon_sw + col/n

Invertida, en continuo (sin redondear, para conservar la parte fraccionaria):

    row = n · (lat_sw + 1 − lat)
    col = n · (lon − lon_sw)

`row0, col0 = floor(...)`; `frow, fcol` = partes fraccionarias = pesos
bilineales. `row1 = min(row0+1, n)`, `col1 = min(col0+1, n)` — clamp para no
salirse del array cuando el punto cae justo en el borde N/E del tile (ahí
`frow`/`fcol` son 0, así que el clamp no distorsiona nada).

Decisión: la interpolación NO cruza al tile vecino; se queda clamped dentro
del tile actual. Esto es EXACTO, no una aproximación: al ser 3601×3601
(n=3600), el tile incluye ambos bordes (fila/columna 0 y fila/columna n), así
que cualquier punto dentro del rango `[lat_sw, lat_sw+1] × [lon_sw, lon_sw+1]`
tiene sus 4 vecinos ya dentro del propio tile. El clamp (`min(row0+1, n)`) solo
entra en juego justo en el borde, donde el peso de la celda que "faltaría" es
0 — no afecta al resultado. Además los tiles SRTM adyacentes duplican la
fila/columna de borde (el borde este de un tile = borde oeste del siguiente),
así que cruzar daría exactamente el mismo valor. No hay motivo para cargar un
segundo tile.

## Voids (-32768) en la interpolación bilineal

Regla de CLAUDE.md: "interpolar de vecinos o propagar `None`, jamás usar el
número". Implementación:

1. Pesos bilineales normales de las 4 esquinas: `w00=(1-frow)(1-fcol)`, etc.
2. Cualquier esquina con valor `-32768` se excluye: su peso pasa a 0 (no se
   mezcla el número en la media).
3. Si la suma de pesos restantes > 0: renormalizar dividiendo por esa suma y
   hacer la media ponderada de las esquinas válidas.
4. Caso degenerado (punto casi exacto sobre un nodo void, las otras 3 esquinas
   válidas tienen peso ~0 → renormalización 0/0): caer a media simple sin
   peso de las esquinas válidas presentes.
5. Si las 4 esquinas son void: devolver `None`.

## Tile no descargado

`locate_tile_path` calcula el nombre esperado a partir de `floor(lat)`,
`floor(lon)`. Si `get_tile` no encuentra ese fichero en `dem_dir`, lanza
`TileNotFoundError` con el nombre esperado y las coordenadas — NO devuelve
`None`. Razón: `None` ya significa "void dentro de un tile real"; si el tile ni
existe es un caso distinto (cobertura pendiente, `dem_dir` mal, coordenadas con
un signo invertido) y CLAUDE.md pide explícitamente no ocultar ese tipo de
fallo (sección Diagnóstico: los bugs de este proyecto dan resultados plausibles
pero incorrectos que no fallan visiblemente).

Consecuencia concreta: el test dorado 5 (altitud Peñalara) necesita
`N40W004.hgt`, ausente en `data/`. Decisión del usuario: NO `xfail` (el test
es correcto, solo falta el dato) — se marca
`@pytest.mark.skipif(not Path("data/N40W004.hgt").exists(), reason=...)` con
motivo explícito, y se añade un test equivalente que sí corre contra
`data/N36W005.hgt` (tile de Málaga, SRTM1 3601×3601, ya presente). El usuario
dará las coordenadas y el valor esperado de altitud en el chat antes de
escribir ese test — pendiente de ese dato, no se inventa un valor.
`TileNotFoundError` se lanza igual en el código de producción
independientemente de qué tests existan; nunca se devuelve 0 por un tile
ausente.

## Caché

Dict de módulo `_tile_cache: dict[str, Tile]`, clave = nombre de fichero
(`"N38W004.hgt"`). `get_tile` mira la caché antes de llamar `load_tile`;
`load_tile` no toca la caché (permite testear la carga aislada).

## Tests — añadir a tests/test_golden.py

- Tamaño/resolución: cargar `N38W004.hgt` real de `data/`, comprobar
  `n == 3600`.
- Conversión celda↔coordenada: round-trip en una celda conocida.
- Bilineal: punto exactamente sobre un nodo de rejilla debe devolver el valor
  crudo de esa celda (sin mezclar vecinos).
- Void: fabricar un array pequeño en memoria (no un tile real) con una esquina
  a `-32768` y comprobar que el resultado ignora esa esquina; las 4 a
  `-32768` → `None`.
- Caso 5 del contrato (altitud Peñalara 40.8508, −3.9578 entre 2400 y 2430):
  `@pytest.mark.skipif` condicionado a que exista `data/N40W004.hgt`, con
  motivo explícito en el mensaje de skip. No es `xfail`: el test es correcto,
  solo falta el tile.
- Equivalente que sí se ejecuta ahora, contra `data/N36W005.hgt` (Málaga):
  PENDIENTE — coordenadas y altitud esperada las da el usuario en el chat
  antes de escribir este test.
- Tile ausente: pedir una coordenada sin tile en `data/` (ej. medio del
  océano) y comprobar que salta `TileNotFoundError`.

## Verificación

```
python -m pytest tests/test_golden.py -v
```

---

# Parte 3: src/horizon/ — rayos, visibilidad y barrido de 360°

## Contexto

Tercer módulo. Depende de `geo/` y `dem/`. Aplica la sección nueva de CLAUDE.md
"Roles de cada fuente de datos": la altitud del objetivo viene de OSM y se pasa
como parámetro; el DEM se consulta SOLO para el terreno intermedio del rayo,
porque SRTM subestima cimas por promediado (La Maroma: 2041.7 medido en rejilla
vs 2069 oficial).

## Funciones nuevas en módulos existentes

### `geo/`: fórmula directa (destino dado azimut y distancia)

Hoy `geo/` solo tiene el problema inverso (`haversine_m`, `azimuth_deg`). El
rayo necesita el directo:

    φ₂ = asin( sin φ₁·cos(d/R) + cos φ₁·sin(d/R)·cos θ )
    λ₂ = λ₁ + atan2( sin θ·sin(d/R)·cos φ₁,  cos(d/R) − sin φ₁·sin φ₂ )

```python
def destination_point_deg(lat_deg: float, lon_deg: float,
                          bearing_deg: float, distance_m: float) -> tuple[float, float]
```

Parámetro `bearing_deg`, NO `azimuth_deg_val`: colisionaría visualmente con la
función `azimuth_deg` del mismo módulo. Mismo convenio horario-desde-el-norte,
reutiliza `R_EARTH_M`.

El rayo usa el azimut INICIAL fijo para todas las muestras; no se recalcula por
muestra (la fórmula directa ya sigue el círculo máximo con ese único ángulo).

### `dem/`: interpolación vectorizada

```python
def elevations_m(lat_arr: np.ndarray, lon_arr: np.ndarray,
                 dem_dir: str = "data") -> np.ndarray
    # bilineal sobre arrays; NaN donde las 4 esquinas son void
    # agrupa por tile internamente (indexado numpy por bloque)
    # TileNotFoundError si algún punto cae en un tile ausente
```

Misma lógica de voids que `elevation_m` escalar (excluir esquinas void,
renormalizar pesos) pero con máscaras booleanas. NaN es el equivalente
vectorizado del `None` escalar.

## API de `src/horizon/`

```python
class Visibility(Enum):
    VISIBLE = "visible"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"

@dataclass
class VisibilityResult:
    status: Visibility
    truncated_at_m: float | None = None   # solo si status == UNKNOWN

@dataclass
class HorizonProfile:
    azimuths_deg: np.ndarray     # (1800,) 0.0, 0.2, ..., 359.8
    elevations_deg: np.ndarray   # (1800,) máx. del rayo; NaN si todo el rayo es void
    truncated_at_m: np.ndarray   # (1800,) NaN si llegó a max_km; si no, distancia de corte

def check_visibility(lat_obs_deg, lon_obs_deg, h_obs_m,
                     lat_tgt_deg, lon_tgt_deg, h_tgt_m) -> VisibilityResult

def horizon_profile(lat_deg, lon_deg, h_obs_m, max_km: float = 150.0) -> HorizonProfile
```

`truncated_at_m` como array float con NaN de centinela (no objetos con `None`):
mantiene los tres arrays homogéneos en float64 para que `render/` pinte los
sectores no fiables con `~np.isnan(truncated_at_m)`.

### h_obs_m es la altitud del OJO

Documentar en el docstring de AMBAS funciones: `h_obs_m` es la altitud del ojo
sobre el nivel del mar, no la cota del suelo. Quien llame debe sumar la altura
de la persona (~1.7 m) a la cota del terreno. Pasar la cota del suelo hace que
las muestras cercanas bloqueen espuriamente (el terreno a 30 m estaría a la
misma altura que el observador).

## Tres estados en vez de bool

Decisión del usuario. `UNKNOWN` cumple el mismo criterio que
`TileNotFoundError` en `dem/` — el dato ausente es visible para quien llama, no
se disfraza de resultado válido — pero como valor de retorno, no excepción,
porque con cobertura solo de Andalucía el caso es esperado y frecuente, no
excepcional. Propagar la excepción rompería `horizon_profile` (casi todos los
1800 rayos saldrían de cobertura). Tratarlo como void sería el fallo silencioso
que CLAUDE.md prohíbe.

Void dentro de un tile real es DISTINTO y sí se salta: es un hueco puntual
dentro de datos que existen, no ausencia de datos.

## Regla del máximo acumulado

Misma regla de CLAUDE.md, dos preguntas distintas:

- **`check_visibility`** — "¿algo tapa esta línea concreta?". Calcula
  `elev_tgt = elevation_deg(d_total, h_obs_m, h_tgt_m)` una vez (con la altitud
  OSM). Recorre las muestras `0 < d_i < d_total` en orden creciente:
  - terreno bloquea (`elevation_deg(d_i, h_obs, terreno_i) >= elev_tgt`) →
    `BLOCKED`, corte inmediato;
  - `TileNotFoundError` → `UNKNOWN(truncated_at_m=d_i)`, corte inmediato;
  - `None` (void) → se salta la muestra;
  - fin del rayo sin nada de lo anterior → `VISIBLE`.

  No hace falta lógica extra para "si ya bloqueó antes, el hueco posterior es
  irrelevante": `BLOCKED` corta el bucle, así que si el bloqueo está antes en
  distancia el hueco ni se consulta. El orden del bucle lo resuelve solo.

- **`horizon_profile`** — "¿cuál es el máximo de todo el rayo?". Sin corte
  anticipado posible: un pico lejano puede superar a uno cercano más bajo.

## np.nanmax, no np.max

`elevations_m` devuelve NaN en los voids y `elevation_deg` propaga ese NaN. Un
solo void envenenaría el máximo del azimut entero con `np.max` /
`np.maximum.accumulate`. Se usa `np.nanmax` (comprobando antes
`np.all(np.isnan(...))` para evitar el RuntimeWarning de all-NaN).

Si TODAS las muestras del rayo son NaN, el perfil de ese azimut es **NaN**, no
un número inventado.

## Truncamiento por orden de distancia, no por grupo de tile

Un rayo puede cruzar el tile A, luego el B, y rozar A otra vez en una esquina.
Agrupar por identidad de tile haría que "la primera muestra del grupo ausente"
no fuese la primera muestra ausente en orden de distancia.

Procedimiento: calcular el nombre de tile de cada muestra, localizar el índice
MÍNIMO cuyo tile no existe, truncar ahí y descartar todas las muestras
posteriores aunque pertenezcan a tiles que sí están descargados.

## Rendimiento del barrido

1800 rayos × hasta 5000 muestras = 9M consultas. Bucle Python puro tardaría
minutos. Medidas:

- Vectorizar la fórmula de destino por rayo con numpy (`np.arange(30, max_m, 30)`
  de una vez, no muestra a muestra).
- `dem.elevations_m` vectorizada, indexado numpy por bloque de tile.
- `_tile_cache` de `dem/` ya evita releer disco (los 1800 rayos parten del mismo
  punto y comparten los primeros tiles).
- `np.nanmax` sobre el array del rayo, sin bucle Python.

`check_visibility` NO se vectoriza, y es deliberado: tiene corte anticipado y
en la práctica los bloqueos aparecen en los primeros km. Vectorizar el rayo
entero obligaría a calcular muestras que el corte habría evitado. Además es una
llamada por objetivo, no 1800.

## Tests

Datos ya verificados en esta sesión (ejecución real contra `data/`):

1. **Horizonte marino** (valida de golpe signo de curvatura, convenio de
   elevaciones negativas, máximo acumulado y fórmula de destino):

   Desde (36.730, −4.097) — mar abierto frente a Torre del Mar, comprobado: las
   2706 muestras hasta el truncamiento son todas 0.0 — con `h_obs_m = 10.0`
   mirando al sur (azimut 180.0):

   ```python
   prof = horizon_profile(36.730, -4.097, 10.0)
   idx = índice del azimut 180.0
   assert prof.elevations_deg[idx] == pytest.approx(-0.0947, abs=0.005)
   ```

   Solución cerrada: maximizando f(d) = (−h − 0.87·d²/(2R))/d sale
   d = √(2Rh/0.87) = 12 102.1 m (12.102 km) y el ángulo −0.094688°. Medido con
   muestreo de 30 m: −0.094688° a d = 12 090 m (la muestra más cercana al
   óptimo; la función es plana en el máximo). Si la curvatura tuviera el signo
   cambiado saldría positivo; si faltara, −0.0473°; si el muestreo estuviera
   mal, no llegaría al máximo.

   El mismo test comprueba `truncated_at_m[idx] ≈ 81180` (no NaN): el rayo pide
   `N35W005.hgt`, que no está descargado. El máximo se alcanza a 12 km, mucho
   antes, así que el valor sigue siendo válido — justo el caso que justifica
   separar "máximo alcanzado" de "hasta dónde llegué".

2. **Rayo que cruza un void** (punto 1 de las correcciones). Los voids reales de
   `data/` están TODOS en la columna 3600 (borde este) de los tiles W001/W002, y
   son inalcanzables vía `elevation_m`: `floor` enruta esa longitud exacta al
   tile siguiente al este, que no está descargado → sale `TileNotFoundError`, no
   void. Verificado tile por tile en esta sesión.

   Por tanto el test usa un **tile sintético** escrito en `tmp_path`: formato
   SRTM3 (1201×1201 = 2 884 802 bytes, ~2.9 MB, mucho más ligero que los 26 MB
   de SRTM1) en una zona sin cobertura real, con una franja de `-32768`. Además
   ejerce el camino SRTM3 de `load_tile`, hoy sin test. Dos asertos:
   - rayo que cruza la franja void pero también terreno válido → el perfil NO es
     NaN (las muestras válidas siguen contando);
   - rayo íntegramente dentro de la franja void → el perfil ES NaN.

3. **Tri-estado de `check_visibility`**, todo con datos disponibles, desde
   (36.730, −4.097, 10.0) mirando al sur:
   - objetivo a 20 km a altitud 0 → `BLOCKED` (el propio mar lo tapa: el
     horizonte marino está a −0.0947° y el objetivo a −0.107°);
   - objetivo a 20 km a 500 m → `VISIBLE` (+1.33°);
   - objetivo a 100 km a 2000 m → `UNKNOWN` con `truncated_at_m ≈ 81180`
     (nada lo bloquea antes; el rayo se queda sin cobertura).

4. **Caso 6 del contrato** (Peñalara → Bola del Mundo visible):
   `@pytest.mark.skipif` sobre `data/N40W004.hgt`, igual que el caso 5. El test
   es correcto, solo falta el tile.

## Verificación

```
python -m pytest tests/test_golden.py -v
```

Más una comprobación manual de que `horizon_profile` completa el barrido de
1800 rayos en segundos, no minutos (no como aserto de test: depende de la
máquina).

---

# Parte 4: validación externa contra PeakFinder

## Contexto

Hasta ahora todos los tests dorados validan funciones aisladas o casos con
solución cerrada calculados por nosotros mismos. `peakfinder_referencia.csv`
es distinto: son 72 cimas con azimut y distancia producidos por una
implementación INDEPENDIENTE (PeakFinder). Valida el motor completo de punta a
punta — `geo` (azimut y distancia) + `dem` (lectura del terreno) + `horizon`
(barrido y máximo acumulado) — contra un tercero que no comparte nuestro
código ni nuestros errores. Ningún test actual hace eso.

El CSV se versiona: es texto (~4 KB), no un tile. `.gitignore` excluye
`data/*.hgt` pero no toca `tests/data/`, así que no hay que cambiarlo.

## Mover el fichero

Hoy está en la raíz del repo sin trackear. Va a
`tests/data/peakfinder_referencia.csv` (`git add` incluido).

Columnas: `nombre, ele_m, dist_km, lat, lon, azimut_deg`. 72 filas de datos.

## El observador NO está en el CSV — se deriva

Problema: el CSV da azimut y distancia pero no desde dónde. Sin el punto de
observación no hay test posible.

Derivado así (verificado en esta sesión): retrocediendo desde cada cima por su
azimut inverso y su distancia, las 72 filas convergen a (36.7446, −4.0902)
±0.0003°. Refinando por búsqueda en rejilla para minimizar el peor residuo de
azimut se llega a **(36.745, −4.090)**, con error máximo de **0.085°** sobre
las 72 cimas y RMS 0.040°. Las coordenadas redondas confirman que es el punto
real, no un artefacto del ajuste.

Se fija como constante documentada en el test:

```python
# Derivado del propio CSV: es el único punto que reproduce los 72 azimuts
# (peor caso 0.085°). No viene en el fichero.
PEAKFINDER_OBS = (36.745, -4.090)
```

Dos parámetros libres contra 72 observaciones con residuos de 0.04° no pueden
absorber un error sistemático (signo invertido, lat/lon cambiados): el ajuste
no vuelve circular al test.

Altura del ojo: cota DEM del observador (3.0 m) + 1.7 = 4.7 m. Comprobado que
el criterio de elevación es insensible a este valor (el margen peor se queda en
−0.072° para alturas de ojo de 4.7 a 50 m), porque al subir el ojo bajan por
igual el ángulo requerido y el perfil.

## Los dos asertos

Para las 72 cimas:

1. **Azimut** calculado con `geo.azimuth_deg` a menos de **0.5°** del de
   PeakFinder, comparado en aritmética modular (`(a-b+180) % 360 - 180`) para
   que el cruce por 0/360 no dé un falso fallo.
   Medido: peor caso 0.085°, holgura de 6x.

2. **Perfil del horizonte** en el azimut de cada cima ≥ elevación de la cima
   **−0.2°**. La elevación requerida se calcula con `geo.elevation_deg` a
   partir de nuestra distancia haversine y la `ele_m` oficial del CSV.
   Medido: peor déficit −0.072° (Benthomiz), 0 de 72 fallan, holgura de ~3x.

## Los márgenes NO son todos del mismo signo (corrección medida)

Medido sobre las 72 cimas, con `margen = perfil − requerido`:

- **41 negativos**, todos diminutos: el peor es **−0.072°**.
- **31 positivos**, y algunos grandes: hasta **+2.517°**.
- media +0.196°, mediana −0.006°.

Los dos signos tienen causas distintas y ninguna es un fallo:

**Negativos** — sesgo conocido de SRTM, documentado en CLAUDE.md: el DEM
subestima las cimas por promediado, así que el perfil queda un poco por debajo
de la altitud oficial. Están acotados: ninguno pasa de −0.072°, muy lejos del
umbral de −0.2°.

**Positivos** — estructurales, no ruido: el perfil es el MÁXIMO de todo el
rayo, mientras que el requerido es la elevación de UNA cima concreta. Una cima
de primer plano puede tener detrás, casi en el mismo azimut, otra más alta que
domina el perfil. Verificado en los dos casos extremos:

- Cerro del Tablón (824 m, 14.6 km, az 19.87°) → el perfil da 5.68° porque
  detrás está Cerro la Cuna (1630 m, 17.6 km, az 19.55°).
- Cerro Juan María (576 m, 9.6 km, az 18.18°) → el perfil da 5.763°, que
  coincide con Cerro Tacita de Plata (1893 m, 18.5 km, az 18.55°, elev 5.77°).

## Qué debe decir el comentario para el diagnóstico futuro

El criterio es unilateral (`perfil ≥ requerido − 0.2°`) precisamente porque los
dos signos significan cosas distintas. El comentario debe dejar escrito:

- Un margen **positivo, por grande que sea, es normal** y no indica bug: es una
  cima más alta detrás en el mismo azimut. No hay cota superior esperable.
- Un margen **negativo está acotado a ~0.1°** por el sesgo de SRTM. Si algún
  día aparece un déficit negativo grande (más allá del −0.2° de tolerancia),
  NO es el sesgo conocido: es un bug distinto — el rayo no llega a la cima, el
  azimut está desplazado, o el DEM se está leyendo mal.

Es decir, la asimetría a vigilar es la contraria a la que sugiere la
intuición: lo sospechoso es el déficit negativo grande, no el exceso positivo.

## Estructura del test

Fixtures con `scope="module"` para que el barrido de 360° (~4 s) se calcule una
sola vez y lo compartan los dos tests, en vez de dos veces.

```python
PEAKFINDER_CSV = Path("tests/data/peakfinder_referencia.csv")

@pytest.fixture(scope="module")
def peakfinder_rows() -> list[dict]      # csv.DictReader, encoding utf-8

@pytest.fixture(scope="module")
def peakfinder_profile() -> HorizonProfile

def test_peakfinder_azimuts(peakfinder_rows)
def test_peakfinder_perfil_alcanza_las_cimas(peakfinder_rows, peakfinder_profile)
```

Ambos tests acumulan TODOS los fallos y los reportan juntos en el mensaje del
assert, en vez de abortar en la primera cima: si algún día se rompe algo,
importa saber si falla una cima o las 72 (un fallo sistemático y uno puntual
tienen diagnósticos opuestos, ver la sección Diagnóstico de CLAUDE.md).

El comentario de cabecera debe dejar dicho que la referencia procede de una
implementación independiente y que valida el motor completo (geo + dem +
horizon), no una función aislada.

`ele_m` en el CSV está en metros y `dist_km` en kilómetros — es la única parte
del proyecto donde entran km, y solo porque el fichero externo viene así; se
convierte a metros al leerlo.

## Verificación

```
python -m pytest tests/test_golden.py -v
```

---

# Parte 5: recorte de azimut en render/ y en el CLI

## Contexto

El panorama de 360° comprime cada grado en ~4 px: para mirar un sector concreto
(la sierra al norte, o el arco que cubre una foto) hace falta poder ampliar. Se
añade un recorte opcional `az_min`/`az_max` a `render_horizon_png`, que reescala
el eje X al rango y filtra las etiquetas de picos, y se expone en
`python -m peakid panorama`. **Sin recorte nada cambia**: es el requisito que
gobierna el diseño de los ticks (ver abajo).

## Ficheros

- [src/render/__init__.py](src/render/__init__.py) — el recorte.
- [peakid/__main__.py](peakid/__main__.py) — `--az-min` / `--az-max`.
- [tests/test_golden.py](tests/test_golden.py) — tests del desenrollado y del CLI.

## Firma

```python
def render_horizon_png(profile, path, peaks=None, width_px=1600,
                       height_px=520, az_min_deg=None,
                       az_max_deg=None) -> LabelReport | None
```

Ambos o ninguno: pasar solo uno es `ValueError`. Retrocompatible.

## El cruce por 0/360: azimut desenrollado

Toda la complejidad se concentra en dos helpers pequeños y testables, y a
partir de ahí el resto del render no se entera de que hay wraparound:

```python
def _crop_span(az_min_deg, az_max_deg) -> tuple[float, float]:
    """Normaliza a [0,360) y desenrolla: si az_max <= az_min le suma 360.
    340→40 da (340, 400), span 60. az_min == az_max se interpreta como
    vuelta completa desde az_min (span 360), no como sector vacío."""

def _unwrap_deg(az_deg, az_min_deg) -> float:
    """az_deg + 360 si az_deg < az_min. Lleva cada azimut al espacio
    desenrollado del recorte: con az_min=340, el 350 sigue en 350 y el 10
    pasa a 370."""
```

**Rotar los arrays del perfil** para que empiecen en `az_min` es lo que hace
que el resto del código siga sirviendo sin tocarlo:

```python
start = int(np.searchsorted(az, az_min))       # primer índice con az >= az_min
az_u = np.concatenate([az[start:], az[:start] + 360.0])   # monótono creciente
# misma rotación a elevations_deg y truncated_at_m, luego máscara al rango
```

Efecto secundario valioso: un tramo de terreno que cruza el norte, hoy partido
en dos runs por el borde del array, pasa a ser **contiguo**. `_runs`, los
polígonos de silueta y las bandas de void funcionan igual y salen mejor.

`az_step_deg` se calcula del array ORIGINAL (antes de recortar): con un sector
estrecho el array cortado puede quedarse en una sola muestra y `az[1]-az[0]`
reventaría.

Margen de una muestra a cada lado al enmascarar (`[az_min - step, az_max + step]`)
para que la silueta llegue pegada al borde, y `x_at` se clampa a `[x0, x1]`
porque pillow no recorta: sin el clamp el polígono se saldría del marco.

## Escala X

```python
span = az_max_u - az_min
def x_at(az_deg_unwrapped):
    return clamp(x0 + (az_deg_unwrapped - az_min) / span * plot_w, x0, x1)
```

Sin recorte: `az_min=0`, `span=360` — idéntico a la fórmula actual.

## Ticks del eje X (aquí está el requisito de no cambiar nada sin recorte)

- **Posición**: paso adaptativo al span (≤15°→2°, ≤30°→5°, ≤90°→10°,
  ≤180°→20°, resto 45°). Con span 360 da 45°, exactamente la rejilla actual.
- **Etiquetas**: los azimuts cardinales (0/90/180/270, comprobados también en
  su copia +360 para que el 0 aparezca como "N (0°)" en un recorte 340→400)
  llevan siempre su letra. Los ticks no cardinales se etiquetan en grados
  **solo cuando hay recorte** — así el panorama completo conserva su aspecto
  actual (solo N/E/S/O) y el sector ampliado gana la escala que necesita.

## Escala Y

Se autoescala a las elevaciones **del sector recortado**, no del perfil
entero: es el sentido de ampliar. Sale gratis calculando `lo`/`hi` después de
recortar.

## Picos

Se filtran con `_unwrap_deg` al rango antes del bucle de prioridad, así que la
competencia por ranuras ocurre solo entre los del sector — un sector estrecho
etiqueta muchos más picos, que es justo lo que se busca.

`LabelReport` gana un tercer campo `out_of_range: list[str]`: los descartados
por el recorte NO se mezclan con los descartados por colisión, que son cosas
distintas (fuera de encuadre vs. perdió la competencia). Mantiene la promesa
de que nada desaparece en silencio.

## CLI

`--az-min` y `--az-max` en el subcomando `panorama` (no en `peaks`: ahí no hay
eje que recortar). Validación: o los dos o ninguno → mensaje claro y salida 1.
El resumen imprime el sector cuando se usa:
`sector 340°–40° (60° de arco)` y, si hay picos fuera, cuántos.

## Tests

- `_crop_span` / `_unwrap_deg`: el caso 340→40 (span 60, 350→350, 10→370, 100
  fuera), el normal 0→60, y `az_min == az_max` → span 360.
- Render recortado: genera el PNG, tamaño correcto, y `LabelReport` solo con
  picos del sector (los de fuera en `out_of_range`).
- Render recortado que cruza el norte, con el perfil real de Torre del Mar.
- CLI: `--az-min 0 --az-max 60` termina en 0 y crea el fichero; pasar solo uno
  falla con mensaje y sin traceback.

## Verificación

```
python -m pytest tests/test_golden.py -v
python -m peakid panorama --lat 36.745 --lon -4.090 --az-min 340 --az-max 40 -o sector.png
```

Más la inspección visual de un sector estrecho y de uno que cruce el norte,
que es como se verifica este módulo.

---

# Parte 6: align — unificar orientación, quitar el detector, persistir sesión

## Contexto

Cuatro correcciones del usuario sobre la herramienta de alineamiento tras
usarla con fotos reales:

1. Sospecha de que la tira lee píxeles sin orientar (foto 141720.jpg:
   píxeles 4080×3060 + tag Orientation 6 → se muestra 3060×4080).
2. El detector automático de cresta debe DESAPARECER: detecta bordes
   arbitrarios y ese problema es del modelo de segmentación de fase 2, no de
   heurísticas.
3. Si existe un .align.json junto a la foto, reanudar desde él (perder el
   ajuste al reabrir es inaceptable con decenas de fotos).
4. Verificación exportable de la tira ANTES de entregar: el bug llegó al
   usuario porque no había forma de comprobar la tira sin abrir la GUI.

**Hallazgo de la exploración (documentar en el informe final): el fallo NO
se ha podido reproducir, y las dos hipótesis de causa quedan descartadas con
evidencia.**

- Orientación: `photo_np.shape=(4080, 3060, 3)` — el array orientado; la
  réplica exacta de `_draw_strip` sobre 141720.jpg + su JSON da tira
  horizontal correcta.
- Semillas fuera de encuadre (hipótesis del usuario): reproducido con los
  valores exactos de la captura (az 0.0, hfov 67.4, pitch 5.05) → tira
  horizontal y reconocible. Con az=0 la línea NO queda fuera: el barrido es
  de 360° y hay perfil hacia el norte (y la sierra de la foto está casi al
  norte, az real ~25°). y_line válida en las 322 columnas, rango 1972–2134,
  salto máximo entre vecinas 6 px.

Lo que la réplica offline NO cubre es la capa Tk (ImageTk, el canvas, la
interacción con redimensionados de ventana): si el fallo vive ahí, solo se
caza en vivo. De ahí la medida forense del punto 4b.

## Ficheros

- [src/align/__init__.py](src/align/__init__.py)
- [src/align/gui.py](src/align/gui.py)
- [peakid/__main__.py](peakid/__main__.py)
- [tests/test_golden.py](tests/test_golden.py)

## 1. Unificación del origen de píxeles

Función pura nueva en `src/align`:

```python
def load_oriented_photo(photo_path) -> tuple[Image.Image, np.ndarray]:
    # exif_transpose UNA vez; devuelve (imagen, array) del MISMO origen,
    # con aserción array.shape[:2] == (img.height, img.width)
```

La GUI la consume para `self.img` (canvas principal) y `self.photo_np`
(tira): un único punto de carga, ninguna otra ruta abre los píxeles
(`seed_from_exif` abre la foto pero solo lee metadatos EXIF, no píxeles).

Test con JPEG sintético con Orientation=6 y un marcador de contenido (esquina
coloreada): tamaño intercambiado tras cargar, imagen y array con dimensiones
coherentes entre sí, y el marcador en la esquina que corresponde tras la
rotación — se verifica contenido, no solo forma.

## 2. Eliminar el detector automático de cresta

Borrar de `src/align`: `detect_skyline_rows`, `strip_residual`,
`StripResidual`, `_sky_trend`, `_median_filter` y el import de
`sliding_window_view`. **Conservar** `projected_y_per_column`: la tira lo
sigue necesitando para enderezar la banda.

En la GUI: quitar la línea turquesa, el label de residuo
(mediana/rms/cobertura), la tecla `A` y `self.residual`. La tira queda: banda
de ±3° de la foto, enderezada y estirada, con la línea roja proyectada en el
centro. El juicio del encaje lo hace el usuario mirando.

Tests: borrar los seis del detector (`test_strip_residual_*`,
`test_detector_*`) y sus helpers (`_fake_strip`, constantes de color).
Conservar `test_projected_y_per_column` y `test_pitch_positivo_baja_la_linea`
(documenta el convenio de signo de la proyección, sigue vigente), reescribiendo
su comentario, que menciona la tecla A.

## 3. Persistencia de la sesión

En `src/align`:

```python
def load_session(photo_path) -> dict | None
    # el .align.json de la foto si existe y es version 1; None si no
```

Precedencia al abrir con `--photo`:
- parámetros iniciales: `alignment` de la sesión si existe; si no, semillas
  (EXIF/default) como hasta ahora;
- `notes` precargadas de la sesión;
- observador: CLI > sesión > EXIF-GPS, con su `source`;
- **el bloque `seed` original se conserva TAL CUAL al reguardar**: la
  procedencia EXIF de la primera vez es la medida del error de brújula y una
  recarga no debe sobrescribirla.

La GUI recibe un `resume: dict | None`; `_save` escribe el seed heredado
cuando existe. El CLI imprime `sesión anterior cargada: <ruta>` al reanudar.

Tests puros (sin GUI): `load_session` presente/ausente/versión rara; y que un
reguardado con seed heredado conserva la procedencia original.

## 4. Verificación exportable de la tira

Extraer el cálculo del parche de `_draw_strip` a función pura en `src/align`:

```python
def build_strip(photo_np, x_px, y_px, usable, columns_px,
                band_px, strip_h) -> tuple[np.ndarray, np.ndarray]
    # (parche RGB strip_h×len(columns_px), columnas válidas)
```

La GUI la llama con su estado de vista — así lo que se verifica ES lo que se
dibuja, no una réplica. Los tests la usan directamente (la del test de
orientación incluida: sobre la foto sintética orientada, la cresta conocida
debe aparecer en la fila esperada de la tira).

## 4b. Volcado forense en vivo (tecla D)

Como el fallo visto por el usuario no se reproduce offline, la GUI gana una
tecla `D` que vuelca a disco, junto a la foto, EXACTAMENTE lo que acaba de
dibujar: el parche de la tira tal cual salió de `build_strip` (mismo array
que se convirtió en ImageTk) más un .txt con el estado completo (parámetros,
view_x/view_scale, cw/ch, tamaños de imagen y de canvas reales según Tk). La
próxima vez que la tira salga mal, `D` captura el estado y el fallo deja de
ser irreproducible. Coste: ~15 líneas.

## Verificación

```
python -m pytest tests/test_golden.py -v
```

- Generar con `build_strip` la tira de 141720.jpg + su .align.json y MIRARLA
  (Read del PNG): debe ser la franja horizontal de sierra con calima. Si sale
  algo vertical o irreconocible, está mal y no se entrega.
- Smoke de construcción de `_AlignApp` sin mainloop (foto sintética
  orientación 6): canvas y tira construidos, sin residuos del detector.
- Reanudación: crear JSON, reabrir estado inicial = alignment guardado.

---

# Parte 7: límites del buscador — saturación, ambigüedad y acotación

## Contexto

Caso real que rompe el buscador: teleobjetivo de Sierra Nevada desde Granada
(Veleta a azimut ~125°, FOV real ~15-20°). La búsqueda devolvió **az 20.1° y
FOV 72.5°**: sitio equivocado del panorama y campo saturado en el borde de la
rejilla.

Son DOS fallos distintos y conviene no confundirlos:

1. **El azimut verdadero nunca estuvo en el espacio explorado.** La rejilla
   cubre ±20° alrededor de la semilla; con semilla 0 (sin brújula EXIF), 125°
   quedaba fuera. El buscador no "se equivocó": nunca miró ahí.
2. **El FOV se saturó en el borde.** Rango 40-75 con óptimo en 72.5 = el
   verdadero (~15-20°) está fuera y la búsqueda se apoya en el extremo.

Y por debajo hay un tercer problema, el más serio, que el usuario identifica
bien: **con FOV estrecho la firma del horizonte es pobre y la correlación
tiene muchos máximos casi equivalentes**. Aunque el rango cubriera el valor
correcto, el buscador podría elegir otro con puntuación casi idéntica. Un
resultado así no debe presentarse como solución.

## Estado ya aplicado (antes de reentrar en modo plan)

En [src/align/search.py](src/align/search.py) ya están escritas las
constantes nuevas (`FOV_RANGE_DEG = (10, 80)`, `FOV_GRID_RATIO`,
`AMBIGUITY_AZ_DEG`, `AMBIGUITY_MARGIN`) y los dataclasses `SearchCandidate`
(sin cambios) y `SearchResult`. Falta todo lo demás.

## 1. Rango de FOV 10-80 con rejilla geométrica

Rejilla **geométrica** (ratio 1.05, ~44 valores de 10 a 80) en vez de lineal:
con paso lineal de 2.5°, el paso es un 3% a 75° pero un 17% a 15° — justo
donde más precisión hace falta. La geométrica da resolución RELATIVA
constante, que es lo que importa al estimar escala.

## 2. Detección de saturación

Tras la etapa gruesa, comprobar si el óptimo se apoya en un extremo:

- `fov_at_edge`: el FOV óptimo es el primer o el último valor de la rejilla.
- `az_at_edge`: el azimut óptimo está a menos de un paso del extremo del
  rango explorado.

Ambos van en `SearchResult` y el CLI los reporta como aviso explícito: el
verdadero puede estar FUERA de lo explorado, con la sugerencia concreta
(`--fov-hint`, `--az-hint`).

## 3. Detección de ambigüedad

De la lista gruesa (ya ordenada por error en píxeles), localizar el mejor
candidato cuya separación en azimut respecto del mejor supere
`AMBIGUITY_AZ_DEG` (10°) — una hipótesis genuinamente distinta, no un vecino
de rejilla. Puntuar AMBOS con la métrica exacta (`_score_exact`) y calcular:

```
ambiguity_margin = (error_alternativa − error_mejor) / error_mejor
ambiguous = ambiguity_margin < AMBIGUITY_MARGIN
```

`AMBIGUITY_MARGIN = 0.25` es un valor de partida **que hay que calibrar
midiendo**: sobre 141720.jpg (panorama ancho, firma rica, resultado que sí
convence) el margen debe salir cómodamente por encima del umbral. Si no,
ajustar el umbral con ese dato y documentarlo. No fijar el número sin medir.

Cuando `ambiguous`, el CLI lo dice explícitamente, muestra la alternativa, y
**no presenta el resultado como solución** — sigue precargando el mejor en la
GUI (es mejor punto de partida que nada) pero avisando de que la búsqueda no
distingue entre ambos.

`SearchResult.reliable` resume las tres reservas.

## 4. Acotación: hints

CLI en el subcomando `align`:

- `--az-hint GRADOS` + `--az-margin GRADOS` (defecto 20)
- `--fov-hint GRADOS` + `--fov-margin GRADOS` (defecto 10)

Con `--fov-hint` la rejilla pasa a lineal fina dentro de
`[hint−margen, hint+margen]` (paso ~rango/24, mínimo 0.5°): cuando el usuario
sabe el campo aproximado, interesa resolución absoluta, no relativa.

El CLI imprime SIEMPRE el espacio explorado (`azimut 105-145°, FOV 12-28°`)
para que un fallo por "nunca miré ahí" sea visible sin adivinar.

## 5. Coste

La rejilla crece ~3× (44 FOV × 81 az × 13 roll ≈ 46 000 evaluaciones frente
a las 15 800 actuales, que tardan 7 s). Mitigación: la etapa gruesa muestrea
**200 columnas** en vez de 400 (la fina sigue con 400). Medir el tiempo real
y, si pasa de ~25 s sin hints, reducir el paso de roll a 1.0° en la gruesa.

## Cambio de API

`search_alignment` devuelve `SearchResult` en vez de `list[SearchCandidate]`.
Actualizar `peakid/__main__.py::_search_alignment` y el test del caso 7
(`test_busqueda_recupera_desplazamiento_sintetico`, que hoy indexa la lista).

## Tests

- Caso 7 del contrato: adaptado a `.candidates`, sin cambiar sus tolerancias.
- **Saturación**: búsqueda con `fov_hint` cuyo margen deja el FOV verdadero
  fuera → `fov_at_edge` True. Y con el rango completo conteniéndolo → False.
- **Ambigüedad**: perfil sintético PERIÓDICO (misma cresta repetida cada
  ~30° de azimut) con FOV estrecho → `ambiguous` True y `alternative` a más
  de 10°. El mismo perfil rico del caso 7 con FOV ancho → `ambiguous` False.
  Este test es el que codifica el fallo real del usuario.
- **Hints**: `az_hint` desplaza el rango explorado (`az_range_deg` lo
  refleja) y encuentra un desplazamiento que sin hint quedaría fuera de ±20°.

## Verificación

```
python -m pytest tests/test_golden.py -v
```

- Medir sobre 141720.jpg: margen de ambigüedad y `reliable` (debe ser fiable,
  es el caso bueno) — es el dato con el que se calibra `AMBIGUITY_MARGIN`.
- Medir el tiempo de búsqueda sin hints y con hints.

---

# Parte 9: las teclas 1..5 no seleccionan candidato

## Contexto

El usuario no puede elegir entre los candidatos de la búsqueda. Causa
confirmada empíricamente en este entorno (Tk 8.6):

```
root.bind('<1>', ...)  ->  registrado como  '<Button-1>'
root.bind('<0>', ...)  ->  registrado como  '0'   (tecla)
root.bind('<Key-1>')   ->  registrado como  '1'   (tecla)
```

En Tk, **un detalle numérico en un binding se interpreta como número de
BOTÓN del ratón, no como tecla**. `<1>`…`<5>` son los cinco botones (1-3 más
la rueda), así que:

1. Pulsar los dígitos 1..5 no hace absolutamente nada — el síntoma que
   reporta el usuario.
2. Peor: **hacer clic con el botón izquierdo dispara `_apply_candidate(0)`**.
   El lienzo tiene su propio `<Button-1>` para desplazar, y las bindtags
   propagan el evento al toplevel, así que arrastrar la foto además repone
   los parámetros del primer candidato. Es un efecto secundario destructivo
   que se lleva el ajuste manual en curso.

`<0>` (volver al encuadre) funciona hoy de milagro: 0 no es un número de
botón válido, así que Tk lo trata como tecla.

## Ficheros

- [src/align/gui.py](src/align/gui.py), en `_bind_events`.

## Cambios

1. `<Key-1>` … `<Key-5>` en lugar de `<1>` … `<5>`.
2. `<Key-0>` en lugar de `<0>`: hoy funciona por accidente, y depender de que
   0 no sea un botón válido es exactamente el tipo de detalle que se rompe
   solo. Deja el fichero sin ningún binding numérico ambiguo.
3. Comentario en el sitio explicando la regla de Tk, para que nadie vuelva a
   escribir `<3>` pensando que es una tecla.
4. Realimentación cuando aún no hay candidatos: pulsar 1..5 sin haber hecho
   una búsqueda debe decirlo en la barra de estado ("aún no hay candidatos:
   pulsa S para buscar") en vez de no hacer nada, que es indistinguible de
   un binding roto — precisamente la ambigüedad que ha costado este rato.

## Verificación

```
python -m pytest tests/test_golden.py -q
```

Más una prueba dirigida, sin abrir ventana, que es la que habría cazado esto:
construir `_AlignApp`, comprobar que `root.bind()` NO contiene `Button-1`
entre los bindings del toplevel, y que `event_generate('<KeyPress-2>')`
cambia el candidato activo mientras que `event_generate('<ButtonPress-1>')`
no lo toca.

---

# Parte 10: fase 2 — segmentación de cielo con modelo

## Contexto

El detector heurístico se rompe con cielo cubierto. Medido sobre
`IMG_20240210_122140.jpg` (9248×6944): se engancha a los **bordes de las
nubes**, solo toca la loma real en el centro-izquierda, y hasta pica en los
árboles del fondo del valle. Desviación típica de las filas detectadas:
**1070 px**, con máximo en 5372. La cobertura dice 78%, pero de basura — otra
muestra de que la cobertura sola no mide calidad.

La causa es estructural, no un umbral mal puesto: el criterio "el cielo es
azul, claro y poco saturado" deja de discriminar cuando las nubes son más
oscuras que el cielo y tienen bordes más marcados que la cresta. Ningún
ajuste de constantes arregla eso; hace falta un modelo que sepa qué es cielo.

## Evidencia recogida (todo verificado en esta sesión)

- 5 alineamientos manuales en `Dataset/*.align.json`: 141720, 141721,
  20260812_104438, Nerja, sierra. Cubren FOV de 40° a 75°, observadores de
  4.7 m a 2882 m de altura, y tamaños de 2551×1701 a 9248×6944.
- **`IMG_20240210_122140.jpg` NO tiene alineamiento guardado**: el caso que
  hay que resolver no está en el conjunto de evaluación. Decisión pendiente
  del usuario (ver abajo).
- `onnxruntime` no está instalado.

## ALCANCE APROBADO: solo programación dinámica, sin dependencias nuevas

Decisión del usuario. El fallo observado es que el detector **salta de la
cresta a una nube y vuelve, porque decide cada columna por separado**. La DP
fuerza un camino continuo y puede resolverlo sola. Primero se mide; solo si
queda margen claro se valora el modelo. La DP no es trabajo desechable: se
usará igual encima del modelo si este acaba entrando.

## 1. Programación dinámica sobre la evidencia actual

El skyline es un camino continuo de la primera a la última columna. Se
resuelve como camino de coste mínimo, todo con numpy.

**Mapa de evidencia** `s(r,c) ∈ [0,1]` (probabilidad de que el píxel sea
cielo). Se obtiene reutilizando el barrido descendente adaptativo que ya
está validado en `_adaptive_scan`, con un cambio: hoy "bloquea" la columna
en cuanto encuentra la cresta; para el mapa hay que recorrerla ENTERA,
congelando la referencia de cielo en el momento del bloqueo y siguiendo
hacia abajo. Sale una desviación por píxel `d(r,c) ≥ 0` respecto al cielo
esperado, y `s = 1 / (1 + exp(k·(d − t)))`.

Es importante conservar ese barrido adaptativo y no volver a un modelo
global de cielo: ya se midió que con calima el degradado no es lineal y un
umbral global dispara cientos de filas por encima de la sierra.

**Coste de que la frontera pase por (r,c)**: cielo mal explicado arriba más
terreno mal explicado abajo,

    coste(r,c) = media(1−s por encima de r) + media(s por debajo de r)

Se calcula para todas las filas de golpe con sumas acumuladas por columna,
O(H·W).

**Transición con penalización de salto**:

    mejor(c,r) = coste(r,c) + min_{|r'−r| ≤ K} [ mejor(c−1,r') + λ·|r−r'| ]

y retroceso para reconstruir el camino. Coste `W × H × K`: con la imagen
submuestreada (el detector ya reduce a ~1200 de ancho; para 9248×6944 sale
1156×868) y K≈25 son ~25M operaciones vectorizadas, del orden de lo que ya
hace el barrido de horizonte.

Esto es exactamente lo que impide el salto cresta→nube→cresta: el camino ha
de ser continuo y el atajo por las nubes paga la penalización de los dos
saltos, aunque su coste local sea menor.

**Validez por columna**: la DP siempre devuelve un camino completo, así que
`valid` deja de significar "encontré cresta" y pasa a significar "el camino
aquí es fiable". Criterio: margen entre el coste del camino y el del mejor
camino alternativo que pase lejos en esa columna, más el filtro de tramos
cortos que ya existe. Sin esto la DP daría 100% de cobertura siempre,
incluida la basura, y perderíamos la señal que hoy avisa de que la foto no
es utilizable.

## 2. Arquitectura: detectores intercambiables

Se conserva la firma actual `detect_photo_skyline(photo_np) -> (cols, rows,
valid)` de [src/align/search.py](src/align/search.py) y se añade un
selector:

- `heuristica` — la actual, SIN TOCAR (es la referencia de comparación).
- `dp` — misma evidencia de color, camino global.

Módulo nuevo `src/align/skyline.py` con el mapa de evidencia y la DP;
`search.py` solo elige. Bandera `--skyline-detector` en el CLI y en la GUI,
para poder comparar sobre la misma foto sin reiniciar.

## 3. Modelo: APLAZADO, con la investigación ya hecha

No se implementa ahora. Queda documentado para no repetir el trabajo:

- Candidato: **SegFormer-B0 finetuneado en ADE20K**
  (`nvidia/segformer-b0-finetuned-ade-512-512`), clase `sky` = índice 2.
- **Aviso**: el DeepLabV3+ de `torchvision` está entrenado en COCO/VOC-21,
  que **no tiene clase cielo**. Para "sky" hay que ir a pesos ADE20K o
  Cityscapes; no vale el modelo que trae torchvision de serie.
- Dependencia: `onnxruntime`, **nunca torch**. Tamaños reales de PyPI
  medidos en esta sesión:

  | paquete | Windows | Linux |
  |---|---|---|
  | onnxruntime 1.28 | **14.1 MB** | 19.2 MB |
  | torch 2.13 | 122.3 MB | **526.6 MB** |

  Torch es 9× mayor en Windows y 27× en Linux, y arrastra torchvision, para
  ejecutar una sola función de inferencia.
- Si entra: opcional (respaldo heurístico si falta), el `.onnx` en `models/`
  y en `.gitignore` como los tiles `.hgt`, y la exportación desde torch se
  hace una vez en un entorno desechable.

## 4. Evaluación

Arnés en `scripts/eval_skyline.py` (NO en tests: depende de fotos que no
están versionadas), que recorre los `Dataset/*.align.json` y reporta por
foto y detector:

**Métrica directa — aísla el detector** (la que de verdad mide lo que
cambiamos): con los parámetros del alineamiento MANUAL fijos, proyectar el
perfil y medir el residuo por columna contra la cresta detectada. Se reporta
mediana en px y en grados, y **fracción saturada**. Es necesaria porque la
métrica de la búsqueda recorta a 40 px y satura: ya nos hizo informar como
"38 px, casi bueno" un ajuste que estaba a 350 px.

**Métrica de punta a punta — la que pidió el usuario**: error angular del
azimut que devuelve `search_alignment` frente al azimut manual, más
cobertura de columnas válidas.

**Robustez**: fracción de columnas cuyo residuo supera 3× la mediana —
mide justamente "se ha ido a las nubes".

Con 5-6 fotos la potencia estadística es baja; el criterio de éxito no es
una media sino que **ninguna empeore** y que el caso de las nubes se
arregle. Se informa foto a foto, no un promedio que escondería un empeora-
miento compensado por otra.

El usuario va a alinear a mano `IMG_20240210_122140.jpg` para que entre como
sexto caso, el más duro. El arnés recorre los `.align.json` que existan, así
que funciona con cinco y con seis sin tocarlo.

## Verificación

```
python -m pytest tests/test_golden.py -q          # nada se rompe
python scripts/eval_skyline.py                    # tabla comparativa
```

- Tests con evidencia sintética, sin fotos: la DP prefiere el camino
  continuo frente a uno con dos saltos aunque el atajo tenga coste local
  menor — el caso cresta→nube→cresta reducido a su esencia.
- Test de que la penalización λ hace lo que dice: con λ=0 la DP degenera en
  la decisión por columna independiente (debe reproducir el mínimo local),
  y al subirla el camino se aplana.
- Inspección visual de la cresta detectada sobre `IMG_20240210_122140.jpg`
  con los dos detectores, que es donde el fallo se ve a simple vista.
- El detector heurístico actual NO se toca: si alguno de sus tests cambia de
  resultado, es que se ha tocado sin querer.

---

# Parte 8: GUI — acotar por sector y resolver pitch/roll en forma cerrada

## Contexto

El ajuste manual es impracticable por dos razones que se suman: cuatro
parámetros acoplados, y un slider de azimut que recorre 360° en ~1100 px, o
sea **~0.33°/px** (un píxel de ratón mueve más que toda la tolerancia útil).
Dos cambios lo arreglan:

1. **Acotar el sector**: panel superior con el perfil de 360°; el usuario
   arrastra el arco aproximado que cubre la foto. Con 30° de rango el slider
   pasa a **~0.03°/px**, diez veces más fino que la precisión que buscamos.
2. **Pitch y roll automáticos**: fijados azimut y FOV, se resuelven en forma
   cerrada contra las columnas con detección fiable y se recalculan en cada
   cambio. Quedan dos parámetros que tocar en vez de cuatro.

Nota de coherencia con la Parte 6: allí se quitó el detector de cresta de la
GUI por engañoso. Vuelve ahora, pero es **otro** detector (el adaptativo de
`search.py`, validado sobre foto real) y con otro papel: alimenta un ajuste
cerrado que el usuario puede congelar, no un número de calidad que invitaba a
confiar. La fiabilidad se hace visible (nº de columnas usadas) y si no hay
detección suficiente el modo automático se desactiva y lo dice.

## Ficheros

- [src/align/search.py](src/align/search.py) — `solve_pitch_roll` (pura).
- [src/align/gui.py](src/align/gui.py) — panel de sector, modo automático.
- [tests/test_golden.py](tests/test_golden.py).

## 1. `solve_pitch_roll` — forma cerrada, modelo YA VERIFICADO

Modelo de primer orden, comprobado numéricamente contra `project_profile`
antes de escribirlo aquí:

```
y(pitch, roll) − y(0, 0)  ≈  f·tan(pitch) + (x − W/2)·roll_rad
```

Es decir: **el residuo vertical contra la línea con pitch=roll=0 es una RECTA
en x**. Su ordenada da el pitch y su pendiente el giro. Medido con
`f = (W/2)/tan(hfov/2)`:

| pitch, roll reales | recuperado |
|---|---|
| +2.0, 0.0 | +2.001, −0.001 |
| 0.0, +1.5 | −0.000, +1.489 |
| +3.0, −2.0 | +3.001, −2.009 |
| −4.0, +3.0 | −4.024, +2.943 |
| +8.0, +5.0 | +7.988, +5.109 |

```python
def solve_pitch_roll(azimuths_deg, elevations_deg, params,
                     skyline_cols_px, skyline_rows_px, width_px, height_px,
                     iterations=2) -> tuple[float, float] | None
```

Procedimiento: proyectar con el pitch/roll actuales, calcular el residuo en
las columnas con detección fiable, rechazar los que se aparten más de 3·MAD
de la mediana (un arbusto o un tejado no debe torcer la recta), ajustar
`residuo = a + b·(x − W/2)` por mínimos cuadrados, y acumular
`pitch += atan(a/f)`, `roll += degrees(b)`. **Se itera dos veces**: la tabla
de arriba es de primer orden y a 8°/5° deja ~0.1° de error; iterando, el
residuo del modelo desaparece. Devuelve `None` si quedan menos de 20 columnas
utilizables.

Los valores se acotan a los límites de los sliders antes de aplicarse.

Nota: el buscador podría usar esta misma función para dejar de rejillar el
giro (7 valores menos por combinación). No se toca ahora — cambiaría los
resultados de la búsqueda y sus tests; queda anotado como mejora aparte.

## 2. Panel de sector

Canvas nuevo de ~110 px arriba del todo (`side="top"` ANTES del canvas de la
foto, que ya se empaqueta con el orden importando — ver Parte 6). Dibuja el
perfil completo con `_runs`-style polilínea sobre los arrays de
`HorizonProfile`, ticks en N/E/S/O, y el sector seleccionado como rectángulo
translúcido.

- **Interacción**: arrastrar con el botón izquierdo marca `[inicio, fin]`.
- **Cruce por el norte**: si el arco seleccionado supera **180°**, se
  interpreta como el COMPLEMENTARIO (el que cruza el norte). No es ambiguo:
  el FOV máximo del proyecto es 80°, así que una selección de 340° solo puede
  significar "los 20° de enfrente". El panel rotula siempre el sector
  resultante (`sector 350°–10° (20°)`), así que no hay sorpresa.
- **Efecto sobre los sliders**: azimut pasa a `[centro − ancho/2,
  centro + ancho/2]` (el propio sector) y FOV a `[ancho·0.5, ancho·2]`,
  acotado a `FOV_RANGE_DEG`. El valor actual se reencaja dentro del rango
  nuevo. La etiqueta del slider muestra la resolución conseguida
  (`azimuth_deg  0.03°/px`) — es el dato que motivó todo esto.
- Sin sector marcado, los rangos son los de ahora (comportamiento actual).

## 3. Búsqueda acotada al sector

`--search` en el CLI corre ANTES de que exista la ventana, así que un sector
elegido en la GUI no puede alimentarlo. Interpretación (la única coherente):
**`--search` arma la búsqueda; marcar un sector la dispara**, acotada a él —
`center_az_deg` = centro, `az_margin_deg` = ancho/2, `fov_hint_deg` = ancho,
`fov_margin_deg` = ancho/2.

Sin `--search`, marcar el sector solo fija los rangos; la tecla `S` lanza la
búsqueda a demanda.

Con el sector el espacio se reduce mucho (medido en Parte 7: 7.7 s con
pistas), pero bloquea el mainloop. Se muestra `buscando…` en la barra de
estado con `update_idletasks()` antes de arrancar, y al terminar se aplican
los parámetros del mejor candidato y se reportan sus reservas
(ambigüedad/saturación) en la misma barra.

## 4. Modo automático en la interfaz

Dos casillas, `auto` junto a `pitch_deg` y a `roll_deg`, **marcadas por
defecto**. Marcada: el slider queda deshabilitado y se actualiza en cada
recálculo. Desmarcada (congelado): el usuario lo arrastra y el solver no lo
toca.

`_redraw` recalcula pitch/roll automáticos ANTES de proyectar, solo si alguna
casilla está marcada. La cresta detectada se calcula UNA vez al abrir
(`detect_photo_skyline`, ~0.3 s) y se guarda.

Si la detección da menos de 50 columnas fiables: casillas desmarcadas y
deshabilitadas, y la barra de estado lo dice ("sin cresta detectable: pitch y
roll quedan manuales"). Nunca se resuelve con datos insuficientes en
silencio.

La barra de estado muestra el nº de columnas usadas en el último ajuste.

## Tests

- **`solve_pitch_roll` recupera pitch y roll conocidos** desde una cresta
  sintética generada con `_foto_desde_perfil`, con tolerancia 0.05°. Es el
  test que ata los signos: si el de roll se invirtiera, el ajuste automático
  torcería la línea al revés sin que nada fallara.
- `solve_pitch_roll` con menos de 20 columnas fiables → `None`.
- Robustez: añadir 10% de columnas con residuo disparatado y comprobar que la
  recuperación sigue dentro de tolerancia (el rechazo por MAD).
- **Sector → rangos**: función pura `sector_to_ranges(start, end)` que
  devuelve `(az_lo, az_hi, fov_lo, fov_hi)`; casos normal, cruce por el norte
  (arco > 180° → complementario) y clamp a `FOV_RANGE_DEG`.
- Smoke de `_AlignApp` con el panel nuevo: se construye, el sector por
  defecto es None, y aplicar un sector cambia los rangos de los sliders.

## Verificación

```
python -m pytest tests/test_golden.py -v
```

- Sobre 141720.jpg: aplicar el sector equivalente a los hints de la Parte 7
  (az 26±10 → sector 16–36°) y comprobar que `solve_pitch_roll` con el azimut
  y FOV del ajuste guardado devuelve un pitch cercano al manual (+6.4).
- Exportar con PIL la misma polilínea que dibuja el panel de sector y MIRARLA:
  debe ser el perfil de 360° reconocible, con el sector marcado donde toca.
  El panel no se puede capturar de Tk, pero su geometría sí se verifica.
