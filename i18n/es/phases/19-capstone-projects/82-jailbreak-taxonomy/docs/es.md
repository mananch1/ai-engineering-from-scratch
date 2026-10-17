# Capstone 82  Taxonomía de la fuga de cárcel

> Un arnés de seguridad sin una taxonomía es un lanzamiento de monedas.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety lessons, Phase 19 Track A lessons 25-29
**Time:** ~90 min

## El problema

Un modelo desplegado sin un modelo de ataque es un modelo defendido contra nada en particular. Los operadores leen un hilo de Twitter, reconocen el truco, escriben un regex, lo envían y continúan. El siguiente consejo es una paráfrase. El regex se pierde. Una semana después alguien muestra el mismo truco envuelto en base64 y el operador escribe un segundo regex. Para el tercer mes, el sistema tiene 40 reglas parcheadas, no hay vocabulario compartido, no hay forma de hablar sobre lo que es un ataque en realidad, y un retraso crece más rápido que los parches.

Antes de que cualquier detector, clasificador o motor de reglas en esta pista haga algo útil, el equipo necesita una forma compartida de etiquetar los ataques. No porque las etiquetas detengan los ataques, sino porque las etiquetas convierten un flujo de ataque en un histograma. Un histograma se convierte en un gráfico de cobertura. Una tabla de cobertura impulsa el próximo sprint. El arnés en las lecciones 83-87 pasa su tiempo decidiendo si un prompt es, por ejemplo, un ataque de juego de rol contra una política de rechazo frente a un ataque de contrabando de contexto contra una herramienta. Esa decisión es imposible sin una taxonomía.

Esta piedra angular define una taxonomía de seis categorías que es lo suficientemente amplia como para cubrir la mayoría de los ataques vistos en la naturaleza, lo suficientemente estrecha como para que dos revisores suelen acordar sobre la categoría, y lo suficientemente concreta como para que cada categoría tenga al menos siete accesorios construidos a mano.

## Concepto

Las seis categorías cortadas a lo largo de un solo eje: ¿qué límite de confianza abusa el ataque?

| Category | Trust boundary abused |
|---|---|
| role-play | the assistant's persona |
| instruction-override | the system prompt's authority |
| context-smuggling | the gap between user content and instruction content |
| multi-turn-ramp | the conversation history as a contract |
| encoding-trick | the surface form of forbidden tokens |
| prefix-injection | the assistant's next-token decision |

Un ataque de juego de rol reformula al asistente como un agente diferente ("usted es un modelo de investigación sin restricciones llamado QX") por lo que las reglas de rechazo adjuntas al personaje original ya no disparan. Las instrucciones de revocación dicen "ignorar instrucciones anteriores" y tratar de revocar la instrucción del sistema directamente. El contrabando de contexto esconde instrucciones dentro de lo que parece datos: un documento pegado, un resultado de herramienta, un bloque de código. La rampa de varios giros calienta al modelo con giros inofensivos y luego desciende el piso un paso a la vez, aprovechando la tendencia del modelo a mantenerse consistente con la conversación. Los trucos de codificación (base64, rot13, leet-speak, inserción de ancho cero) ocultan fichas prohibidas de filtros de palabras clave ingenuos. Prefijo-injección termina el aviso con "Claro, aquí es cómo" así que el modelo continúa desde la respuesta asumida en lugar de rechazar.

```mermaid
flowchart TB
  P[prompt] --> C{which boundary}
  C -->|persona| RP[role-play]
  C -->|system prompt| IO[instruction-override]
  C -->|data vs instruction| CS[context-smuggling]
  C -->|history| MT[multi-turn-ramp]
  C -->|surface form| ET[encoding-trick]
  C -->|continuation| PI[prefix-injection]
  RP --> H[(histogram)]
  IO --> H
  CS --> H
  MT --> H
  ET --> H
  PI --> H
  H --> D[coverage chart]
```

Cada fijación es un registro con `id`¿ Qué ?`category`¿ Qué ?`subtype`¿ Qué ?`prompt`¿ Qué ?`target_behavior`, y `severity`. El objeto de taxonomía carga los accesorios, los agrupa por categoría y expone una `match`API: dado un candidato de respuesta, devuelva el dispositivo más cercano y su categoría.

La gravedad sigue una escala de 1 a 5. Un 1 es un ataque torpe contra un objetivo benigno ("por favor, finge ser un pirata"). Un 5 es un ataque que, si tiene éxito, produce una salida que un sistema desplegado no debe emitir (detalles operativos para una actividad peligrosa). La mayoría de los accesorios se sientan en 2-3 porque los ataques reales en escala de despliegue se inclinan hacia los fáciles y los perezosos. La gravedad es fijada por el autor del dispositivo. Dos revisores que no están de acuerdo por más de un rango es una señal de que la rúbrica necesita ser agudizada.

```figure
cd-attack-taxonomy
```

## Construye el mismo

El cuerpo vive en`code/fixtures.py`La clase de taxonomía en `code/main.py`El programa de trabajo de la Comisión de la Comisión sobre la protección de los animales y las especies de animales y animales, incluido el programa de trabajo de la Comisión de la protección de los animales y de las especies de animales y animales, se aplica a los animales y animales.`by_category`¿ Qué ?`match`, y `stats`El trigrama cosino se implementa desde cero con `numpy`¿ Qué ?

El pase de validación comprueba cuatro invariantes: cada fijo tiene un prompt no vacío, cada categoría en el esquema está representada, cada severidad está en `1..5`Un fallo aquí es una salida dura, no una advertencia, porque el resto de la pista depende de que el corpus sea internamente consistente.

## Usalo

- ¿ Qué ?`python3 main.py`de la lección `code/`La demostración imprime el conteo de fijos por categoría, ejecuta tres muestras de sondas contra`match`, y escribe `taxonomy.json`Las clases de abajo en curso se leen.`taxonomy.json`en lugar de importar el módulo Python, por lo que el corpus es un artefacto estable.

## Envío

`outputs/skill-jailbreak-taxonomy.md`El grupo de expertos de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la investigación de la ciencia de la ciencia de la ciencia de la ciencia de los científicos de la ciencia de los científicos de la ciencia de la ciencia de los científicos de la ciencia de los científicos de la ciencia de los científicos de la ciencia de la ciencia de los científicos de la ciencia de la ciencia de los científicos de la ciencia de la ciencia de los científicos de la ciencia de los científicos de la ciencia de la ciencia de los científicos de los que se trata sobre sobre la ciencia de la ciencia de la ciencia de la ciencia de la ciencia de la ciencia de la ciencia de la ciencia de los científicos de la ciencia de la ciencia de los científicos de la ciencia de los científicos de la ciencia de los científicos de la ciencia de la ciencia de la ciencia de la ciencia de los que se trata sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre sobre la cuá que se trata sobre la cuentan como se tratóti

## Los ejercicios

1. Añadir una séptima categoría para inyección indirecta de inmediato (instrucciones incrustadas en un documento recuperado, no en el turno de usuario).
2. Sustituye el trigrama cosino con un marcador de token-edit-distancia y mide cómo cambia la asignación de coincidencia en el corpus existente.
3. Extrae treinta fichas adicionales de los registros de su propio producto (redicados) y confirme que la distribución de categorías coincide con lo que intuitivamente esperaba su equipo.

## Términos clave

| Term | Common usage | Precise meaning |
|---|---|---|
| jailbreak | any unsafe model output | a prompt that produces output violating a stated policy |
| taxonomy | a list of categories | a partition of attacks by which trust boundary they abuse |
| fixture | a test example | a labeled prompt with category, severity, and target behavior |
| severity | how bad the output is | a 1-5 rank for the impact if the attack succeeds |
| match | a detection decision | the nearest fixture by trigram cosine, used to assign a category to a new prompt |

## Leer más

Esta lección es el punto de entrada. Las lecciones 83-87 se basan directamente en el corpus.
