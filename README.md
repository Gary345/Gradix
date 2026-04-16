# Gradix

Gradix es una aplicacion de vision por computadora para **pre-grading visual de cartas Pokemon TCG**. El proyecto transforma una fotografia en un conjunto de metricas tecnicas y visuales que ayudan a estimar la calidad de captura, rectificar la carta, extraer features y generar una prediccion supervisada de condicion.

El enfoque de Gradix no busca reemplazar un servicio profesional de grading. Su objetivo es ofrecer una **evaluacion preliminar, reproducible y explicable** que sirva como apoyo para analisis, organizacion de inventario y exploracion academica.

## Alcance actual

- Carga y analisis de imagenes desde una interfaz en Streamlit.
- Deteccion del contorno exterior de la carta.
- Rectificacion por perspectiva y validacion post-warp.
- Extraccion de features visuales, geometricas, de centrado, bordes, esquinas y superficie.
- Scoring heuristico intermedio para apoyar la interpretacion del estado de la carta.
- Prediccion binaria `damaged` / `undamaged` con un modelo supervisado integrado en la app.
- Notebooks para documentacion academica del pipeline, dataset y modelado.

## Pipeline real del proyecto

El flujo principal implementado en el repositorio sigue este orden:

1. Carga de imagen.
2. Deteccion del contorno exterior de la carta.
3. Warp de perspectiva para rectificar la captura.
4. Validacion post-warp.
5. Extraccion de features.
6. Calculo de scores heuristicas.
7. Construccion de dataset tabular.
8. Inferencia del modelo de condicion.

Las piezas centrales de ese pipeline estan en:

- `app.py`: interfaz principal en Streamlit.
- `src/pipeline/card_analysis.py`: orquestacion del analisis por imagen.
- `src/vision/card_detector.py`: deteccion del contorno exterior.
- `src/vision/perspective.py`: rectificacion y warp.
- `src/vision/postwarp_validation.py`: validacion de la carta rectificada.
- `generar_dataset.py`: procesamiento por lote para construir el dataset.
- `src/services/condition_model.py`: carga del modelo y prediccion final.

## Estructura del repositorio

```text
Gradix/
|-- app.py
|-- generar_dataset.py
|-- requirements.txt
|-- notebooks/
|-- src/
|   |-- pipeline/
|   |-- vision/
|   |-- features/
|   |-- scoring/
|   |-- services/
|   |-- ui/
|   |-- utils/
|-- data/
|   |-- raw/
|   |-- processed/
|-- model_v3/
```

## Requisitos

- Python 3.11+ recomendado
- Dependencias listadas en `requirements.txt`

Instalacion sugerida:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecucion local

Para levantar la aplicacion:

```bash
streamlit run app.py
```

## Datos y artefactos

El proyecto ya incluye artefactos utiles para trabajo local:

- `data/raw/`: imagenes fuente organizadas por clase.
- `data/processed/dataset_gradix.csv`: dataset tabular generado por el pipeline.
- `model_v3/condition_model_hgb_v3.pkl`: modelo supervisado integrado.
- `model_v3/feature_names.json`: lista ordenada de variables usadas por el modelo.
- `model_v3/metrics.json`: metricas de evaluacion del modelo.

## Notebooks

La carpeta `notebooks/` concentra el material de apoyo y documentacion del proyecto:

- `01_dataset_y_pipeline_gradix.ipynb`
- `01_dataset_autosuficiente_gradix.ipynb`
- `01_dataset_pipeline_fuerte_gradix.ipynb`
- `02_modelado_y_evaluacion_gradix.ipynb`
- `gradix_notebook_final.ipynb`

## Consideraciones

- El target `damaged` / `undamaged` es operacional y depende de la organizacion del dataset.
- La calidad de captura afecta directamente la estabilidad del pipeline.
- Algunas integraciones de servicios externos pueden requerir configuracion adicional por variables de entorno.
- El proyecto debe interpretarse como una herramienta de apoyo y no como certificacion profesional.

## Estado del proyecto

Gradix se encuentra en una etapa funcional de MVP avanzado: el pipeline principal, la generacion de dataset y la inferencia del modelo ya estan integrados, pero el sistema sigue siendo perfectible en robustez, validacion experimental y cobertura de casos reales.
