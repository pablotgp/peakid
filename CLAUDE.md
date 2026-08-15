# PeakID — identificación de cimas por geometría

## Qué es esto

App que, sabiendo dónde estás y hacia dónde apuntas la cámara, dice qué
montañas se ven y cómo se llaman. La identificación es GEOMÉTRICA, no
por reconocimiento de imagen. Se calcula qué debería verse desde una
posición usando un modelo digital de elevaciones (DEM), y se cruza con
una base de datos de topónimos.

El único punto donde entra ML (fase posterior) es segmentar la línea
cielo/terreno en la foto para corregir el error de la brújula. Nunca
para identificar picos.

## Estado

Fase 1: motor de geometría en Python. Sin app móvil, sin ML todavía.

---

# CONVENCIONES — NO NEGOCIABLES

Estas convenciones son la principal fuente de bugs del proyecto. Un
error de signo aquí produce resultados plausibles pero incorrectos que
no fallan de forma visible.

## Coordenadas

- Siempre `(lat, lon)` en ese orden. NUNCA `(lon, lat)`.
- Grados decimales. Norte positivo, Este positivo.
- Altitudes en metros sobre el nivel del mar.
- Distancias en metros dentro del código. Los km solo aparecen en
  mensajes al usuario y en la fórmula de curvatura (documentada abajo).

## Ángulos

- Azimut: grados desde el norte geográfico, sentido HORARIO, rango [0, 360).
  Norte = 0, Este = 90, Sur = 180, Oeste = 270.
  NO es el convenio matemático (desde el eje X, antihorario).
- Elevación: grados sobre el plano horizontal del observador.
  Positivo hacia arriba. Puede ser negativo (mirar hacia un valle).
- Toda función pública recibe y devuelve GRADOS. Radianes solo dentro
  del cuerpo de la función.
- Sufijo obligatorio en los nombres: `azimuth_deg`, `elevation_deg`,
  `lat_rad`. Una variable de ángulo sin sufijo es un bug esperando.
- Normalizar siempre con `az % 360.0` al devolver un azimut.
- **AL PORTAR, ese `% 360.0` NO se traduce literalmente.** En Python el resto
  toma el signo del DIVISOR, así que `-10.0 % 360.0` da `350.0`. En C, C++,
  Java, Kotlin, Swift, JavaScript y Rust toma el signo del DIVIDENDO y da
  `-10.0`. La traducción literal produce azimuts NEGATIVOS en todo el
  cuadrante noroeste, y son plausibles porque las cuentas intermedias siguen
  saliendo. Medido al portar a Kotlin: el caso dorado 2 está a 336°, que por
  esa vía sale −24°. Aplica igual al `(x + 180) % 360 - 180` de
  `destination_point_deg` y de `project_profile`. El puerto usa
  `normalizeAzimuthDeg` / `wrapDeltaDeg`, y ningún punto suyo usa `% 360.0` a
  pelo. Ninguno debe.

## Curvatura terrestre y refracción

SIEMPRE aplicadas. No existe ninguna función de visibilidad o de
elevación en este proyecto que no las incluya.

    R = 6_371_000.0        # radio terrestre medio, metros
    k = 0.13               # coef. de refracción atmosférica estándar
    drop_m = (1 - k) * d_m**2 / (2 * R)

Atajo equivalente con d en km:  drop_m ≈ 0.0683 * d_km**2

Valores de referencia: 10 km → 6.8 m | 30 km → 61.5 m | 60 km → 245.9 m

A 60 km esto son 246 metros. Ignorarlo hace que aparezcan cimas que en
realidad están ocultas.

## Fórmulas de referencia

Azimut inicial (great circle):

    θ = atan2( sin(Δλ)·cos(φ₂),
               cos(φ₁)·sin(φ₂) − sin(φ₁)·cos(φ₂)·cos(Δλ) )

Distancia: haversine sobre R = 6_371_000 m.

Ángulo de elevación del objetivo B visto desde A:

    elevation_deg = degrees(atan2(h_B − h_A − drop_m, d_m))

Inclinación y giro a partir del residuo (sostiene `solve_pitch_roll`, la
etapa gruesa del buscador y el modo automático de la GUI):

    y(pitch, roll) − y(0,0)  ≈  f·tan(pitch) + (x − W/2)·roll_rad
    con f = (W/2) / tan(hfov/2)

Es decir: el residuo vertical entre la cresta detectada y la línea proyectada
es una RECTA en x. Su ordenada da la inclinación y su pendiente el giro, así
que ambos se resuelven en forma cerrada en vez de buscarse. Verificado
numéricamente contra la proyección: recupera pitch a 0.02° y giro a 0.1°.

## Cámara y alineamiento

- `h_obs_m` es la altitud del **OJO** sobre el nivel del mar, no la cota del
  suelo. Quien llame suma la altura de la persona (~1.7 m). Pasar la cota
  hace que las muestras cercanas bloqueen espuriamente.
- Límites FÍSICOS de cámara: inclinación **±30°**, giro **±15°**. El de
  inclinación estuvo en 10° y era demasiado estrecho: fotografiar una cima de
  2500 m desde un valle a 9 km exige mirar 14.5° hacia arriba. El giro sin
  acotar hacía que la búsqueda devolviera −30°, un grado de libertad falso
  con el que el ajuste se retuerce hasta encajar ruido.
- El rayo de visibilidad se detiene **200 m antes del objetivo**: a esa
  distancia el "terreno" ES la ladera del propio objetivo, y sin ese margen
  una cima se bloquea a sí misma por centésimas de grado.
- **Un único origen de píxeles**: `load_oriented_photo` aplica
  `exif_transpose` UNA vez, y el tamaño se lee DESPUÉS — al rotar 90° se
  intercambian ancho y alto, y de eso depende toda la proyección. Ninguna
  otra ruta abre los píxeles de la foto.
- Un arco de sector **mayor de 180° se interpreta como el complementario**
  (el que cruza el norte). Es inequívoco SOLO porque el FOV máximo son 80°:
  si ese tope subiera por encima de 180°, la regla deja de valer.
- **Límite de salto de la DP: `jump_limit = max(25, 2·banda)`, DESATADO de la
  banda a propósito.** Son dos mecanismos con trabajos distintos y no deben
  compartir valor: la **banda** (±2 cuantos alrededor de la frontera del
  modelo) es la que impide vagar hacia las nubes; el **tope de salto** solo
  suaviza el camino DENTRO de ella. Atados —`jump_limit = min(25, banda)`—
  una pared vertical sale en diagonal, porque el camino no puede seguir a la
  banda cuando es la banda la que salta. Medido con el Naranjo de Bulnes
  (`Urriellu_desde_el_Pozo_de_La_Oracion.jpg`): el salto máximo quedaba
  estrangulado en 5 filas, exactamente el tope, con un 2.25% de columnas
  contra él; desatados sube a 13, en línea con las 15 de la heurística, y la
  pared del picu sale vertical.
- Para juzgar un detector hace falta ver la cresta DETECTADA, no solo la
  silueta proyectada: en la GUI con la tecla **C**, y aislada de toda
  proyección con `scripts/dump_skyline.py`. Con un mal encaje, la línea
  proyectada mezcla "el detector se fue a las nubes" con "el azimut está a
  20°", que son problemas distintos y solo el segundo se arregla desde la
  ventana. Ambas vistas dibujan **también las columnas descartadas**: un
  detector con 30% de cobertura parece impecable si solo se pintan las
  válidas.

## Ficheros SRTM (.hgt)

- SRTM1 (1 arcsec): 3601×3601 valores `int16` BIG-ENDIAN (`>i2`).
  Tamaño exacto: 25 934 402 bytes. Verificar al abrir.
- SRTM3 (3 arcsec): 1201×1201. Soportar ambos deduciendo del tamaño.
- Fila 0 = borde NORTE del tile. Columna 0 = borde OESTE.
  (Es el orden inverso al de la latitud: fila creciente = lat decreciente.)
- El nombre del fichero indica la esquina SUROESTE:
  `N40W004.hgt` cubre lat 40..41, lon −4..−3.
- Conversión celda → coordenada, con `n = size - 1`:
      lat = lat_sw + 1 − row / n
      lon = lon_sw + col / n
- Valor `-32768` = dato ausente (void). NUNCA tratarlo como altitud.
  Interpolar de vecinos o propagar `None`, jamás usar el número.
- Interpolación bilineal para consultas entre celdas.
- Cachear los tiles abiertos en memoria (`mmap` o dict). Un barrido de
  360° hace millones de consultas.

## Muestreo del terreno

- Paso de 30 m a lo largo del rayo (≈ resolución del SRTM1).
- Distancia máxima: 150 km. Más allá casi nunca hay visibilidad real
  y multiplica el coste.
- Barrido de horizonte: pasos de 0.2° de azimut (1800 rayos).
- Un punto es visible si su ángulo de elevación supera el máximo
  acumulado de todos los puntos anteriores del mismo rayo.

## Picos (OpenStreetMap)

- Fuente: Overpass API, `https://overpass-api.de/api/interpreter`
- Consulta: `node["natural"="peak"](around:100000, LAT, LON);`
- Tags útiles: `name`, `ele`, `wikidata`. `ele` falta a menudo → usar
  la altitud del DEM como respaldo.
- `name` lleva el topónimo LOCAL, que en Asturias, Galicia, Euskadi o
  Catalunya no es el castellano. Leer también `alt_name` y `name:es` y
  rotular ambos: el Naranjo de Bulnes está como `name=Picu Urriellu` con
  `alt_name=Naranjo de Bulnes`, y parecía faltar de la lista.
- Las coordenadas de OSM están puestas a ojo y pueden desviarse decenas
  de metros. Antes de comprobar visibilidad, RECOLOCAR cada pico en el
  punto más alto del DEM dentro de un radio de 200 m.
- El tag `prominence` casi nunca está relleno. No depender de él;
  filtrar por distancia y por `ele` mientras no se calcule prominencia
  propia.
- Cachear las respuestas de Overpass en disco. La API tiene rate limit
  y no debe consultarse en cada ejecución de los tests.

---

# TESTS DORADOS

`tests/test_golden.py` es el contrato del proyecto.

**NO modificar los valores esperados de este fichero sin aprobación
explícita del usuario en el chat.** Si un test dorado falla, el bug
está en el código, no en el test. Ajustar el valor esperado para que
pase es la peor cosa que se puede hacer en este repo.

Casos (tolerancia ±2% salvo indicación):

1. Curvatura: 10 km → 6.8 m | 30 km → 61.5 m | 60 km → 245.9 m

2. Puerta del Sol (40.4168, −3.7038, 650 m) → Peñalara (40.8508,
   −3.9578, 2428 m):
   - distancia ≈ 52.9 km (±1 km)
   - azimut ≈ 336° (±1°)   ← noroeste; si sale otra cosa, hay un
     signo o un orden lat/lon invertido
   - elevación CON curvatura ≈ 1.72° (±0.05°)
   - elevación SIN curvatura ≈ 1.93°  (test de control: si el valor
     principal da 1.93, falta la corrección)

3. Simetría: azimut(A→B) y azimut(B→A) difieren en 180° ±0.5°.

4. Sentido: avanzar con azimut 0° aumenta la latitud y deja la
   longitud casi igual. Con 90°, aumenta la longitud.

5. DEM: la altitud en (40.8508, −3.9578) está entre 2400 y 2430.
   Si sale ~800, las filas están invertidas.
   Se mide TRAS RECOLOCAR al máximo del DEM dentro de 200 m, igual que hace
   el motor con toda coordenada de cima: esa coordenada (de Wikipedia) queda
   a 175 m de la cumbre real, y leída literalmente da 2391.2 m — 37 m por
   debajo de los 2428 oficiales — frente a 2424.2 recolocada, déficit de
   solo 4 m, coherente con el sesgo conocido de SRTM.

6. Visibilidad: desde el Peñalara, la Bola del Mundo (40.7906,
   −3.9553, 2265 m) es visible. Están a ~7 km sin obstáculos.

7. Sintético: renderizar el horizonte desde un punto, desplazarlo
   artificialmente +13.7°, y comprobar que el alineamiento recupera
   13.7° ±0.1°. (Implementado y en verde.)

## Qué vigila cada caso, y qué NO

Medido con mutaciones al portar a Kotlin, no deducido. **El convenio de
`atan2` —el orden de sus argumentos, que es la sospecha número 1 de la lista
de diagnóstico— lo vigila UN SOLO caso dorado: el 2, por su valor de 336°.**

- El **caso 3 (simetría) NO lo vigila.** Intercambiar los argumentos de
  `atan2` refleja el azimut al convenio matemático, y la propiedad
  ida/vuelta = 180° se conserva intacta bajo esa reflexión. El test pasa
  igual de verde con el convenio girado.
- El **caso 4 (sentido) TAMPOCO lo vigila.** Solo ejercita
  `destination_point_deg` y no llama a `azimuth_deg` en ningún momento, así
  que una mutación en el azimut le es invisible.

Un único guardián para el convenio más peligroso del proyecto es poco. El
segundo, independiente, es **ida y vuelta**: avanzar con
`destination_point_deg` a un rumbo conocido y comprobar que `azimuth_deg`
recupera ese mismo rumbo. Cruza las dos funciones, así que caza el `atan2`
cambiado Y el `%` de la sección de ángulos. En el puerto está como
`caso4_idaYVueltaRecuperaElPuntoDePartida`; **falta implementarlo en Python.**

---

# TRABAJO

## Reglas

- Toda función nueva en `src/geo/` necesita un test con un caso
  verificable a mano.
- Antes de implementar geometría, exponer el plan y las convenciones
  que se van a usar. Modo plan por defecto.
- Un módulo no se da por terminado hasta que sus tests pasan.
- No acumular varios módulos sin verificar: los errores de este
  proyecto no se manifiestan hasta mucho después.

## Diagnóstico

Cuando un resultado salga raro, el orden de sospecha es:

1. ¿Está girado? (convenio de azimut, atan2 con argumentos cambiados)
2. ¿Está invertido? (filas N/S, signo de longitud oeste)
3. ¿Unidades equivocadas? (grados/radianes, metros/km)
4. Y solo entonces: ¿está mal la fórmula?

Casi siempre es 1, 2 o 3.

## Trampas de medición (sufridas, no teóricas)

- La métrica de error del buscador **recorta cada residuo a 40 px y satura**.
  Un desajuste de 350 px da el mismo número que uno de 45, así que un error
  cercano al tope NO significa "casi bueno" sino "sin medir". Mirar siempre
  `saturated_fraction` antes de interpretar un error.
- El error en píxeles ordena candidatos **dentro de una hipótesis, no entre
  hipótesis**. La comprobación de ambigüedad contrasta el mejor contra UN
  alternativo lejano y no cubre un continuo de óptimos parecidos. Para elegir
  entre hipótesis hay que mirar los **topónimos**: qué cima cae sobre qué
  bulto y con qué altitud. Medido: tres candidatos separados 25° con errores
  decrecientes, y el de MENOR error ponía un cerro de 708 m sobre el macizo
  dominante y dejaba fuera de cuadro una cima de 2069 m.
- Una referencia de alineamiento dudosa NO puede usarse como verdad para
  evaluar detectores: se marca `low_confidence` y queda fuera del criterio.
  Medir contra una referencia mala es peor que no medir, porque un detector
  mejor mediría peor.
- **Lo que descalifica una referencia no es haber USADO la búsqueda, sino
  haberla ACEPTADO SIN REVISAR.** Esta distinción costó una vuelta atrás: el
  primer criterio descartaba toda referencia que hubiera pasado por el
  buscador, y eso tira trabajo humano legítimo. Verificar dónde caen los
  TOPÓNIMOS es información independiente del detector —es el criterio que
  esta misma sección señala como el que zanja entre hipótesis—, así que una
  referencia revisada así no hereda el sesgo aunque partiera de él. Lo que sí
  queda inservible es la salida del buscador tomada tal cual: ahí el detector
  se mide contra su propia respuesta.
- Cada alineamiento guarda un bloque **`provenance`**: `search_used`,
  `auto_pitch_roll`, `detector`, `manual`, `manual_review`, `review_delta`,
  `review_note`, `review_kind` y la conclusión `reviewed`. `review_kind`
  (`correccion` / `verificacion`) se DECLARA, no se deduce de que
  `review_delta` esté vacío: en `141720` la revisión corrigió de verdad
  —cambió de hipótesis entre candidatos por los topónimos— y aun así no dejó
  delta numérico. **Solo las `reviewed` deciden** en
  `scripts/eval_skyline.py`, que separa "revisadas" de "aceptadas sin
  revisar". La GUI marca `manual_review` sola: registra qué parámetros movió
  la persona DESPUÉS de la ayuda automática y cuánto (los cambios del ajuste
  cerrado de pitch/roll no cuentan, porque no pasan por la mano).
- Las cinco referencias de `Dataset/` constan como **revisadas**, con nota de
  cómo se verificó cada una. Dos niveles, y conviene no confundirlos:
  - **revisadas con corrección**: `sierra` (FOV corregido a mano de 78.5 a
    74.7 al ver que las etiquetas no caían sobre las cimas) y `141720` (azimut
    zanjado comparando topónimos entre tres candidatos, no por la métrica).
  - **verificadas sin corrección**: `141721`, `20260812` y `Nerja`. Se
    contrastaron los topónimos y se dieron por buenas sin mover ningún valor.
- **LAS CINCO REFERENCIAS ACTUALES TIENEN SU ORIGEN EN LA HEURÍSTICA**: el
  punto de partida de todas lo produjo la búsqueda automática con ese
  detector. La revisión humana las valida como verdad —por eso deciden—, pero
  **el residuo en píxeles sigue favoreciéndolo por construcción**, sobre todo
  en las tres de `review_kind: verificacion`, donde los valores son
  literalmente los que el buscador propuso. Consecuencia práctica: una
  diferencia de residuo del orden de 1 px en contra de otro detector NO es
  evidencia de que sea peor. Lo que sí es comparable hoy: el error de azimut
  de punta a punta (`search_alignment` reajusta todo) y lo que no depende de
  referencia — cobertura, dispersión de filas e inspección visual.
  **Cuando haya referencias nacidas sin búsqueda, el residuo en píxeles
  volverá a ser una métrica comparable entre detectores.**
- **Un JSON sin `provenance` no decide por su cuenta**, pero tampoco se
  descarta: hay que anotarle la revisión. Deducir la procedencia comparando
  `alignment` con `seed` se probó y **falla**: con una sola magnitud
  comparable el criterio degenera (`141720`), y en `sierra` hubo tres
  correcciones simultáneas, que es justo lo que una regla de "casi todo
  coincide" descarta. La deducción solo informa.
- Para una referencia sin ninguna ayuda automática:
  `peakid align --no-auto-pitch-roll`, sin `--search` y sin pulsar S.

## Stack

- Python 3.11+
- Permitido: `numpy`, `pillow`, `requests`, `pytest`
- NO usar: GDAL, rasterio, pyproj, geopy. Las fórmulas se implementan
  a mano — son cinco líneas y así quedan bajo el control de los tests.
- `onnxruntime`: dependencia **OPCIONAL**, ya adoptada para segmentación de
  cielo. Sin ella el motor no revienta — `build_model_detector` devuelve el
  detector heurístico y lo dice. 14.1 MB (Win) / 19.2 MB (Linux) frente a
  torch 122.3 MB / 526.6 MB. **torch nunca como dependencia de ejecución**:
  el `.onnx` se obtiene una vez aparte y aquí solo se ejecuta.
  En Windows requiere el **Visual C++ Redistributable** (`msvcp140.dll`);
  Python solo trae `vcruntime140*`, así que sin él el import falla con
  `DLL load failed` — se captura como `ModelUnavailable`.
- Sin type checker estricto, pero sí type hints en las firmas públicas.

## Segmentación de cielo (modelo)

- **El detector por defecto es `modelo+dp`, y depende de una dependencia
  OPCIONAL.** Es una excepción deliberada a la regla de que el motor funcione
  con solo `numpy`+`pillow`, y se sostiene sobre dos cosas: lo medido y el
  respaldo. Medido, frente a la heurística:

      foto                    heurística      modelo+dp
      141720.jpg              1.5 px, cob 51%   2.5 px, cob 100%
      141721.jpg              2.2 px, cob 96%   2.5 px, cob 100%
      20260812_104438.jpg     2.2 px, cob 99%   2.8 px, cob 100%
      Nerja                   1.9 px, cob 100%  2.3 px, cob 100%
      sierra.jpg             31.3 px, cob 98%  26.2 px, cob 100%
      IMG_20240210_122140    desv 1070 px       desv 232 px, cob 100%

  Gana en cobertura en las seis, arregla la de cielo cubierto (donde la
  heurística se engancha a los bordes de nube) y empata en error de azimut
  (≤0.25° en todas). Los déficits de 0.3–1.0 px se miden contra referencias
  cuyo punto de partida lo produjo la propia heurística: son revisadas y
  válidas, pero un sesgo residual a su favor a escala de píxel es esperable
  (ver `## Trampas de medición`).
- **El respaldo es lo que hace legítimo ese defecto**: sin `onnxruntime` o sin
  el `.onnx` se usa la heurística y la herramienta funciona igual. El aviso
  se da **una sola vez por máquina** (marca en `~/.peakid/`) o con
  `--verbose`: quien no tenga la dependencia no ha elegido nada y no le falta
  nada, así que un aviso en cada ejecución sería ruido.
- Modelo: **SegFormer-B0 finetuneado en ADE20K**, clase `sky` = **2**. 15.3 MB
  en `models/`, NO versionado (como los `.hgt`). Ojo: el DeepLabV3+ de
  torchvision es COCO/VOC-21 y **no tiene clase cielo**.
- El grafo ONNX tiene alto y ancho **dinámicos**, y eso importa: saca los
  logits a **1/4 de la entrada**, y ese cuanto es el suelo del error de
  localización. Medido, el residuo lo sigue: 8.4 px con entrada 512, 2.0 px
  con 1536, y 2048 no mejora y triplica el tiempo. **1536 en el lado largo,
  conservando la relación de aspecto** y en múltiplos de 32.
- El modelo dice QUÉ es cielo pero no con qué fila exacta; el gradiente de la
  imagen dice la fila pero no sabe qué es una nube. Se combinan: DP sobre
  `región(modelo) + 0.3·borde(imagen)`, **acotada a una banda de ±2 cuantos**
  alrededor de la frontera cruda del modelo. La banda no es un detalle: el
  borde de una nube es tan fuerte como el de una cresta, así que sin acotar
  dónde puede pasar el camino, el término de borde reintroduce exactamente el
  fallo que el modelo vino a resolver. El **tope de salto** de esa DP va
  desatado de la banda (ver `## Cámara y alineamiento`): confundirlos
  estrangula las paredes verticales.

## Estructura

    src/geo/      coordenadas, distancia, azimut, curvatura, elevación
    src/dem/      lectura e interpolación de .hgt
    src/horizon/  rayos, visibilidad, barrido de 360°
    src/peaks/    Overpass, recolocación, filtrado
    src/render/   PNG del perfil del horizonte
    src/align/    alineamiento foto↔horizonte, detectores de cresta

## Roles de cada fuente de datos

- DEM: terreno INTERMEDIO (qué tapa la vista) y silueta del horizonte.
- OSM: posición y altitud OFICIAL de las cimas.
- La visibilidad de un pico se comprueba contra SU altitud de OSM, no la
  del DEM. El DEM se consulta solo para el terreno del camino, que sí
  está bien representado a 30 m.
- Motivo: SRTM subestima cimas por promediado (medido: La Maroma
  2065.6 vs 2069 oficial). En agujas estrechas el error es mayor y
  algunas ni aparecen en la rejilla.

## Pendiente (no implementar aún)

Soportar múltiples fuentes de DEM con resoluciones distintas, con
prioridad a la más fina. Los Dolomitas tienen LiDAR abierto a 1-2 m
(Bolzano/Trento) que no sigue el formato de tile 1°x1°. SRTM queda
como respaldo global.

## Margen de cima

SUMMIT_MARGIN_M = 200. El rayo deja de comprobar obstáculos en los
últimos 200 m antes del objetivo. Mismo valor que el radio de
recolocación de peaks/, y por la misma razón: a esa distancia el
"terreno" y el objetivo son el mismo accidente geográfico.

Sin esto se produce AUTO-BLOQUEO: la ladera final del propio pico,
rendida por SRTM ligeramente por debajo de la cota oficial pero un
poco más cerca del observador, gana el ángulo por centésimas y tapa
su propia cima. Medido: Cima de Tejeda salía BLOCKED por terreno de
2068.3 m situado a 13 m de la cima oficial de 2069.

NO bajar este valor sin reproducir el test dorado de PeakFinder.

`geo/` no depende de nadie. Todo lo demás depende de `geo/`.