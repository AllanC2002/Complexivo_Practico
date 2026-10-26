# Complexivo_Practico — Scoring Crediticio

Proyecto académico de scoring crediticio para estimar la probabilidad de incumplimiento (`estado_default`) a partir de las características de una solicitud de crédito. Incluye preparación y análisis de datos, entrenamiento y evaluación de modelos, una API REST con FastAPI y una interfaz web para consultar métricas y realizar predicciones.

> **Nota:** las bandas de riesgo y las decisiones que muestra la aplicación son demostrativas para el proyecto académico; no representan una política crediticia validada ni deben utilizarse por sí solas para tomar decisiones reales.

## Tecnologías

- Python
- Pandas y NumPy para procesamiento de datos
- Scikit-learn, XGBoost y OptBinning para entrenamiento, evaluación y transformación WoE
- Joblib para guardar y cargar artefactos
- FastAPI y Uvicorn para la API REST
- HTML, CSS y JavaScript para la interfaz web

## Estructura del proyecto

```text
Complexivo_Practico/
├── api.py                         # API de predicción con FastAPI
├── index.html                     # Interfaz web
├── script.js                      # Lógica del frontend y conexión con la API
├── style.css                      # Estilos del frontend
├── requirements.txt               # Dependencias de Python
├── data/
│   ├── raw/                       # Dataset original
│   ├── processed/                 # Dataset limpio
│   ├── reference/                 # Diccionarios y variables de referencia
│   └── results/                   # Comparaciones, métricas y matrices de confusión
├── models/                        # Modelos .joblib y metadata_modelo_final.json
├── reports/
│   └── figures/                   # Gráficos generados durante el análisis
├── docs/                          # Archivos de inventario y documentación auxiliar
├── data.py                        # Generación del dataset sintético
├── limpieza.py                  # Limpieza del dataset
├── correlacion.py                 # Análisis y gráficos
├── export.py                    # Entrenamiento y exportación del modelo final
├── 4models.py                     # Comparación general de modelos
├── model5v.py                     # Evaluación del escenario de 5 variables
├── model6v.py                     # Evaluación de los escenarios de 6 variables
└── model8v.py                     # Evaluación del escenario de 8 variables
```

La carpeta `.git` se conserva para mantener el historial del repositorio. Los scripts de Python se ejecutan desde la raíz del proyecto, salvo que posteriormente se cambie su ubicación.

## Requisitos

- Python instalado.
- Los archivos de datos requeridos dentro de `data/`.
- El artefacto del modelo `.joblib` que la API tenga configurado dentro de `models/`.

## Instalación en Windows (PowerShell)

Abre PowerShell en la carpeta raíz del proyecto y ejecuta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno, ejecuta lo siguiente en esa misma terminal y vuelve a activar el entorno:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Ejecución de la aplicación

La API y el frontend se ejecutan en **dos terminales separadas**. Mantén ambas abiertas mientras uses la aplicación.

### 1. Iniciar la API

Desde la raíz del proyecto:

```powershell
python .\api.py
```

La API utiliza `http://127.0.0.1:8000` y carga el artefacto configurado en `api.py`. Comprueba su estado en:

- Estado de la API: <http://127.0.0.1:8000/health>
- Información del modelo: <http://127.0.0.1:8000/model-info>
- Documentación interactiva: <http://127.0.0.1:8000/docs>

En `/health`, comprueba que `model_loaded` sea `true`.

### 2. Iniciar el frontend

Abre una segunda terminal en la raíz del proyecto y ejecuta:

```powershell
python -m http.server 5500 --bind 127.0.0.1
```

Abre el frontend usando esta dirección:

<http://127.0.0.1:5500/index.html>

El frontend está configurado para comunicarse con la API en `http://127.0.0.1:8000`. Para evitar inconsistencias entre `localhost`, IPv4 e IPv6 durante el desarrollo, utiliza las direcciones `127.0.0.1` indicadas arriba.

## Datos y resultados

Las rutas esperadas son:

| Carpeta | Contenido |
|---|---|
| `data/raw/` | `clientes_solicitud_credito.csv`, dataset original |
| `data/processed/` | `clientes_solicitud_credito_clean.csv`, dataset limpio |
| `data/reference/` | `diccionario_binning_woe.csv`, `iv_seleccion_final.csv` y `variables_modelo_final.csv` |
| `data/results/` | CSV de comparación de modelos, resultados de test y matrices de confusión |
| `models/` | Artefactos `.joblib` y `metadata_modelo_final.json` |
| `reports/figures/` | Figuras del análisis, como la matriz de correlación y la tabla de IV |

El frontend carga las comparaciones de los escenarios de 5 y 8 variables desde `data/results/`. El escenario de 6 variables genera resultados para análisis, incluidos los subescenarios definidos en `model6v.py`.

## Scripts de análisis y entrenamiento

Ejecuta estos scripts únicamente cuando necesites regenerar datos, volver a evaluar modelos o entrenar un nuevo artefacto. No son necesarios para iniciar el frontend y realizar predicciones con el modelo ya exportado.

```powershell
# Opcional: regenerar el dataset original
python .\data.py

# Limpiar los datos originales
python .\limpieza_c.py

# Generar análisis y figuras
python .\correlacion.py

# Evaluar escenarios
python .\model5v.py
python .\model6v.py
python .\model8v.py
python .\4models.py

# Entrenar y exportar el artefacto final configurado en el script
python .\export_c.py
```

Ejecuta los scripts desde la raíz del proyecto para que las rutas relativas coincidan con la estructura descrita. `data.py` puede volver a generar el dataset original; ejecútalo solo cuando realmente quieras regenerarlo. Después de exportar un modelo nuevo, reinicia la API para que cargue el artefacto actualizado.

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Estado general y confirmación de carga del modelo |
| `GET` | `/health` | Estado de salud y estado de carga del modelo |
| `GET` | `/model-info` | Escenario, variables y configuración del modelo cargado |
| `POST` | `/predict` | Calcula la probabilidad de default para una solicitud |

## Solución de problemas

- **La API no carga el modelo:** confirma que el archivo `.joblib` exista en `models/` y que la ruta configurada en `api.py` coincida con su nombre.
- **El frontend no muestra métricas:** verifica que existan `comparacion_modelos_5_variables.csv` y `comparacion_modelos_8_variables.csv` en `data/results/`; recarga el navegador con `Ctrl + F5`.
- **Error 404 al cargar un CSV:** revisa que la ruta del archivo en `script.js` o `index.html` coincida con su nueva ubicación dentro de `data/`.
- **No se puede realizar una predicción:** confirma que la API siga ejecutándose y que <http://127.0.0.1:8000/health> muestre `model_loaded: true`.
- **Faltan dependencias:** activa el entorno virtual y ejecuta `python -m pip install -r requirements.txt`.

## Estado del proyecto

El frontend se sirve localmente y consume la API de FastAPI. El modelo seleccionado se carga desde los artefactos guardados; los scripts de evaluación son independientes de la ejecución cotidiana de la aplicación.
