# AGENTS

## Enfoque actual
- Objetivo principal actual: detectar primero el contorno exterior real de la carta, luego rectificar, luego validar el warp y solo despues calcular features/scoring.
- Mantener el pipeline estable mientras migramos: no romper la funcionalidad existente sin justificarlo.
- Mantener compatibilidad hacia atras en interfaces publicas cuando sea posible.
- Priorizar cambios incrementales, pequenos y auditables.

## Reglas de implementacion
- Usar BGR como formato interno estandar de OpenCV, salvo en fronteras de entrada/salida.
- No mezclar cambios cosmeticos con cambios funcionales en la misma tarea.
- Si una tarea es demasiado grande, dividirla en subpasos antes de editar.
- Cuando se toque deteccion o warp, validar primero el orden del pipeline antes de ajustar features o scoring.

## Entregables por cambio
- Explicar siempre que archivos se cambiaron y por que.
- Explicar siempre que se valido: pruebas, scripts, flujo manual o limitaciones si no se pudo validar.

## Puntos de entrada clave
- `app.py`: flujo interactivo principal en Streamlit.
- `src/vision/card_detector.py`: deteccion del contorno exterior / bbox candidata.
- `src/vision/perspective.py`: rectificacion y warp.
- `generar_dataset.py`: pipeline por lote para extraccion de features y scoring.
