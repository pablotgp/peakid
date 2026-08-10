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

5. DEM: la altitud leída en (40.8508, −3.9578) está entre 2400 y 2430.
   Si sale ~800, las filas están invertidas.

6. Visibilidad: desde el Peñalara, la Bola del Mundo (40.7906,
   −3.9553, 2265 m) es visible. Están a ~7 km sin obstáculos.

7. Sintético (cuando exista el alineamiento): renderizar el horizonte
   desde un punto, desplazarlo artificialmente +13.7°, y comprobar que
   el alineamiento recupera 13.7° ±0.1°.

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

## Stack

- Python 3.11+
- Permitido: `numpy`, `pillow`, `requests`, `pytest`
- NO usar: GDAL, rasterio, pyproj, geopy. Las fórmulas se implementan
  a mano — son cinco líneas y así quedan bajo el control de los tests.
- Sin type checker estricto, pero sí type hints en las firmas públicas.

## Estructura

    src/geo/      coordenadas, distancia, azimut, curvatura, elevación
    src/dem/      lectura e interpolación de .hgt
    src/horizon/  rayos, visibilidad, barrido de 360°
    src/peaks/    Overpass, recolocación, filtrado
    src/render/   PNG del perfil del horizonte

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

`geo/` no depende de nadie. Todo lo demás depende de `geo/`.