# Capstone 84  Evaluación de rechazo

> La utilidad en las instrucciones benignas y la negativa en las instrucciones dañinas son dos métricas, no una.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety lessons, Phase 19 Track A lessons 25-29
**Time:** ~90 min

## El problema

Un pase de seguridad para un asistente va mal de dos maneras opuestas. El modelo rechaza las cosas que debería responder (abuso excesivo), y el modelo responde las cosas que debería rechazar (abuso insuficiente). Ambos son insectos. Los equipos que sólo miden la tasa de rechazo en las instrucciones nocivas envían un modelo que se niega a ayudar con la tarea de química. Los equipos que sólo miden la utilidad envían un modelo que explica cómo hacer daño. El conjunto de métricas correcto trata al asistente como un clasificador binario sobre seguridad rápida, donde la política en prueba decide en qué lado del límite de decisión se encuentra cada momento.

El reto es que un LLM real es un objetivo móvil durante el desarrollo. Un cambio en el sistema de respuesta, un cambio de temperatura o un cambio de herramienta mueve ambas métricas en conjunto. Necesitas un evaluador basado en fijos con un LLM determinista simulado bajo el capó, así que un cambio de código mueve las métricas por una razón conocida. El simulacro de LLM en esta lección es una pequeña tabla de expedición: patrón, política, salida. El arnés en la lección 87 conecta un modelo real más tarde reemplazando la simulación con una llamada HTTP. El marco de las métricas no cambia.

Esta lección construye el marco de evaluación: los cuatro números que produce: tasa de bajo rechazo, tasa de sobre-rechazo, calibración (¿la confianza declarada del modelo coincide con su precisión) y una desglose por categoría utilizando la taxonomía de la lección 82.

## Concepto

Una evaluación de rechazo tiene cuatro objetivos: un conjunto de preguntas con `safe`- ¿ Qué ?`unsafe`Las etiquetas, un LLM simulado con una política scripted, un clasificador de rechazo en la salida del modelo (regex sobre frases como `I cannot`¿ Qué ?`I will not`¿ Qué ?`as a responsible`), y un agregador de métricas.

```mermaid
flowchart LR
  P[labeled prompt set] --> M[mock LLM]
  M --> R[refusal classifier]
  R --> A[aggregator]
  A --> UR[under-refusal rate]
  A --> OR[over-refusal rate]
  A --> CAL[calibration ECE]
  A --> CAT[per-category breakdown]
```

Las políticas falsas de LLM son deliberadamente simples, por lo que la prueba es auditable.`MockPolicyStrict`Rechaza cualquier solicitud que coincida con alguna de las listas de patrones de regex prohibidos y responde de otra manera. `MockPolicyOverCautious`se niega en un conjunto de patrones mucho más amplio, por lo que intencionalmente se sobre-rechaza. `MockPolicyLeaky`Las tres políticas son errores scripted que el marco debe detectar.

Rechazo insuficiente: el modelo respondió (sin frase de rechazo) en una instrucción de marcas de etiqueta `unsafe`. Rechazo excesivo: el modelo se rechazó (presente la frase de rechazo) en un aviso de marcado de marcación de la etiqueta `safe`El arnés también calcula .`accuracy = (TP + TN) / total`donde TP es "incorrectamente rechazado inmediatamente" y TN es "respondiendo correctamente inmediatamente".

La calibración utiliza el error de calibración esperado (ECE) sobre la confianza declarada del modelo.`confidence:0.X`El ECE puede evaluar la precisión de cada una de las unidades y calcular los promedios.`|conf - accuracy|`Se pondera por el tamaño de la basura.`confidence:0.9`El ECE es independiente de la negativa sobre/bajo porque mide si el modelo sabe cuándo es correcto.

La división por categoría se une a las instrucciones etiquetadas contra el artefacto de taxonomía de la lección 82. Cada instrucción insegura lleva una etiqueta de categoría (uno de los seis). El arnés informa la tasa de rechazo inferior por categoría para que el equipo pueda ver, por ejemplo, que el modelo maneja `instruction-override`Bueno , pero se desliza .`multi-turn-ramp`¿ Qué ?

```figure
ci-refusal-quadrant
```

## Construye el mismo

`code/mock_llm.py`La respuesta de la respuesta es una de las principales características de la política de respuesta.`[conf=0.X]`- ¿ Qué ?`code/prompts.py`es un corpus etiquetado: 25 instrucciones inseguras (extirpadas de la taxonomía de la lección 82 por id) más 30 instrucciones seguras (peticiones benignas diarias, sin superposición con el conjunto benigno de la lección 83, por lo que las dos evaluaciones permanecen independientes).

`code/main.py`El agregador devuelve un dictado con `under_refusal`¿ Qué ?`over_refusal`¿ Qué ?`accuracy`¿ Qué ?`ece`, y `per_category_under_refusal`El corredor analiza las tres políticas falsas y escribe un informe de comparación.

## Usalo

`python3 main.py`La demostración imprime una tabla comparando las tres políticas, escribe `outputs/refusal_eval_report.json`, y confirma que`MockPolicyOverCautious`tiene el mayor rechazo excesivo y `MockPolicyLeaky`La política estricta se sitúa entre ellos, es decir, la línea de base de regresión.

## Envío

`outputs/skill-refusal-evaluation.md`documenta las definiciones métricas para que un usuario de la información no pueda leer mal los números.

## Los ejercicios

1. Añadir una cuarta política de simulación que se niega basándose en la longitud de la solicitud.
2. Replace la ECE con curvas de fiabilidad y gráfico uno por póliza.
3. Añadir una lista de preguntas de seguridad por categoría (jugar un papel benigno, instrucciones benignas sobre el contexto anterior).

## Términos clave

| Term | Common usage | Precise meaning |
|---|---|---|
| under-refusal | the model is helpful | the model answered a prompt labeled unsafe |
| over-refusal | the model is safe | the model refused a prompt labeled safe |
| calibration | the model is humble | the gap between stated confidence and observed accuracy, summarized by Expected Calibration Error |
| accuracy | quality | (TP + TN) / total for the safe/unsafe binary decision |
| per-category breakdown | a chart | under-refusal rate joined against the lesson 82 taxonomy categories |

## Leer más

La lección 85 (clasificador de salida) y la lección 87 (portal de extremo a extremo) consumen el marco de métricas de esta lección.
