
import os
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# 1. CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
# Por defecto intenta cargar primero el modelo XGBoost.
DEFAULT_ARTIFACT = BASE_DIR / "modelo_scoring_8_variables_xgboost.joblib"

ARTIFACT_CANDIDATES = [
    DEFAULT_ARTIFACT,
    BASE_DIR / "modelo_scoring_8_variables_regresion_logistica_l2.joblib",
    BASE_DIR / "modelo_lr_produccion.joblib",
]

ruta_configurada = os.getenv("SCORING_MODEL_PATH")

if ruta_configurada:
    ruta_modelo = Path(ruta_configurada)

    if not ruta_modelo.is_absolute():
        ruta_modelo = BASE_DIR / ruta_modelo
else:
    ruta_modelo = next(
        (
            ruta
            for ruta in ARTIFACT_CANDIDATES
            if ruta.exists()
        ),
        DEFAULT_ARTIFACT,
    )

# Umbral predeterminado. Si el artefacto contiene uno,
# se utilizará el guardado durante el entrenamiento.
UMBRAL = 0.5

app = FastAPI(
    title="Motor API Scoring Crediticio",
    version="1.0.0",
    description="API para predecir el riesgo de incumplimiento crediticio.",
)

# CORS para desarrollo local.
# En producción, limita allow_origins al dominio de tu frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 2. VARIABLES DE REFERENCIA
# ============================================================

VARIABLES_8 = [
    "edad_titular",
    "ingreso_mensual_usd",
    "ratio_deuda_ingreso",
    "antiguedad_laboral_meses",
    "consultas_buro_ultimos_6m",
    "meses_maximo_mora_historica",
    "score_buro_externo",
    "monto_credito_solicitado",
]

VARIABLES_5 = [
    "ratio_deuda_ingreso",
    "meses_maximo_mora_historica",
    "score_buro_externo",
    "antiguedad_laboral_meses",
    "consultas_buro_ultimos_6m",
]


# ============================================================
# 3. ESTADO DEL MODELO
# ============================================================

modelo = None
binners = None

variables_guardadas = None
usa_woe_guardado = None
nombre_modelo_guardado = None
escenario_guardado = None

error_carga = None
modo_compatibilidad = False
artefacto_cargado = None


# ============================================================
# 4. CARGA DEL MODELO Y SUS METADATOS
# ============================================================

def cargar_artefactos():
    global modelo
    global binners
    global variables_guardadas
    global usa_woe_guardado
    global nombre_modelo_guardado
    global escenario_guardado
    global error_carga
    global modo_compatibilidad
    global artefacto_cargado
    global UMBRAL

    modelo = None
    binners = None
    variables_guardadas = None
    usa_woe_guardado = None
    nombre_modelo_guardado = None
    escenario_guardado = None
    error_carga = None
    modo_compatibilidad = False
    artefacto_cargado = None

    try:
        if not ruta_modelo.exists():
            raise FileNotFoundError(
                f"No se encontró el artefacto: {ruta_modelo}. "
                "Comprueba el nombre del archivo y la carpeta."
            )

        artefacto = joblib.load(ruta_modelo)
        artefacto_cargado = ruta_modelo.name

        # Caso 1: el archivo contiene un diccionario de metadatos.
        if isinstance(artefacto, dict):
            modelo_guardado = artefacto.get(
                "modelo",
                artefacto.get("model"),
            )

            if modelo_guardado is None:
                raise ValueError(
                    "El archivo es un diccionario, pero no contiene "
                    "una clave 'modelo' o 'model'."
                )

            modelo = modelo_guardado

            variables_guardadas = artefacto.get(
                "variables",
                artefacto.get("features"),
            )

            usa_woe_guardado = artefacto.get(
                "usa_woe",
                artefacto.get("uses_woe"),
            )

            nombre_modelo_guardado = artefacto.get(
                "nombre_modelo",
                artefacto.get("model_name"),
            )

            escenario_guardado = artefacto.get(
                "escenario",
                artefacto.get("scenario"),
            )

            umbral_guardado = artefacto.get(
                "umbral",
                artefacto.get("threshold"),
            )

            if umbral_guardado is not None:
                UMBRAL = float(umbral_guardado)

            # En el artefacto de entrenamiento pueden estar
            # los binners bajo cualquiera de estos nombres.
            binners = artefacto.get(
                "binnings",
                artefacto.get(
                    "binners",
                    artefacto.get("binners_produccion"),
                ),
            )

        # Caso 2: el archivo contiene directamente el estimador.
        else:
            modelo = artefacto
            modo_compatibilidad = True

        if modelo is None:
            raise ValueError("El artefacto no contiene un modelo válido.")

        if not hasattr(modelo, "predict_proba"):
            raise TypeError(
                f"El objeto cargado ({type(modelo).__name__}) "
                "no tiene predict_proba(). Se requiere un clasificador "
                "que permita estimar probabilidades."
            )

        # Buscar los binners por separado, si no estaban incluidos.
        if binners is None:
            posibles_binners = [
                BASE_DIR / "binners_produccion.joblib",
                BASE_DIR / "binners_produccion_5.joblib",
            ]

            for ruta_binners in posibles_binners:
                if ruta_binners.exists():
                    binners = joblib.load(ruta_binners)
                    break

        print("\n" + "=" * 55)
        print("MODELO CARGADO CORRECTAMENTE")
        print("=" * 55)
        print(f"Archivo: {artefacto_cargado}")
        print(f"Tipo: {type(modelo).__name__}")
        print(f"Nombre: {obtener_nombre_modelo()}")
        print(f"Escenario: {obtener_escenario()}")
        print(f"Variables: {obtener_variables_modelo()}")
        print(f"Utiliza WoE: {modelo_usa_woe()}")
        print(f"Umbral: {UMBRAL}")
        print("=" * 55 + "\n")

    except Exception as error:
        modelo = None
        error_carga = str(error)

        print("\nERROR AL CARGAR EL MODELO")
        print(error_carga)
        traceback.print_exc()


# ============================================================
# 5. FUNCIONES DE METADATOS
# ============================================================

def obtener_variables_modelo():
    if variables_guardadas is not None:
        return list(variables_guardadas)

    if modelo is not None:
        nombres = getattr(modelo, "feature_names_in_", None)

        if nombres is not None:
            return list(nombres)

        # Compatibilidad con algunos pipelines de scikit-learn.
        if hasattr(modelo, "steps") and modelo.steps:
            estimador = modelo.steps[-1][1]
            nombres = getattr(estimador, "feature_names_in_", None)

            if nombres is not None:
                return list(nombres)

    # Respaldo para modelos que no guardan los nombres de columnas.
    if "5_variables" in str(escenario_guardado):
        return VARIABLES_5.copy()

    if "8_variables" in str(escenario_guardado):
        return VARIABLES_8.copy()

    if ruta_modelo.name.startswith("modelo_scoring_5"):
        return VARIABLES_5.copy()

    return VARIABLES_8.copy()


def modelo_usa_woe():
    # Los metadatos del artefacto tienen prioridad.
    if usa_woe_guardado is not None:
        return bool(usa_woe_guardado)

    variables = obtener_variables_modelo()

    # Detección por el nombre de las columnas cuando no
    # existen metadatos explícitos.
    if variables and all(
        nombre.endswith("_woe") for nombre in variables
    ):
        return True

    return False


def obtener_nombre_modelo():
    if nombre_modelo_guardado:
        return str(nombre_modelo_guardado)

    nombre_archivo = ruta_modelo.name.lower()

    if "xgboost" in nombre_archivo:
        return "XGBoost"

    if "logistica_l2" in nombre_archivo or "logistica" in nombre_archivo:
        return "Regresion Logistica L2"

    if "modelo_lr_produccion" in nombre_archivo:
        return "Regresion Logistica"

    return type(modelo).__name__ if modelo is not None else "Sin modelo"


def obtener_escenario():
    if escenario_guardado:
        return str(escenario_guardado)

    cantidad = len(obtener_variables_modelo())

    if cantidad == 5:
        return "5_variables"

    if cantidad == 8:
        return "8_variables"

    return f"{cantidad}_variables"


# ============================================================
# 6. MODELO DE ENTRADA
# ============================================================

class Prospecto(BaseModel):
    edad_titular: float = Field(ge=18, le=100)
    ingreso_mensual_usd: float = Field(gt=0)
    ratio_deuda_ingreso: float = Field(ge=0)
    antiguedad_laboral_meses: float = Field(ge=0)
    consultas_buro_ultimos_6m: float = Field(ge=0)
    meses_maximo_mora_historica: float = Field(ge=0, le=12)
    score_buro_externo: float = Field(ge=300, le=850)
    monto_credito_solicitado: float = Field(gt=0)
    tipo_vivienda: str = "propia"


def obtener_datos_cliente(cliente):
    # Compatibilidad con distintas versiones de Pydantic.
    if hasattr(cliente, "model_dump"):
        return cliente.model_dump()

    return cliente.dict()


# ============================================================
# 7. TRANSFORMACIÓN WoE
# ============================================================

def transformar_woe(datos, variables):
    if binners is None:
        raise ValueError(
            "El modelo requiere transformación WoE, pero no se "
            "cargaron los binners. Verifica que el artefacto incluya "
            "'binnings' o que exista binners_produccion.joblib."
        )

    if not isinstance(binners, dict):
        raise ValueError(
            "Los binners cargados no tienen el formato esperado. "
            "Se esperaba un diccionario indexado por variable."
        )

    datos_woe = {}

    for nombre_variable_modelo in variables:
        # El nombre guardado puede ser variable_woe o el nombre original.
        nombre_original = nombre_variable_modelo.removesuffix("_woe")

        if nombre_original not in datos:
            raise ValueError(
                f"No se recibió la variable '{nombre_original}'."
            )

        if nombre_original not in binners:
            raise ValueError(
                f"No existe un binner para '{nombre_original}'. "
                f"Variables disponibles: {list(binners.keys())}"
            )

        binner = binners[nombre_original]
        valor_original = datos[nombre_original]

        try:
            valor_transformado = binner.transform(
                np.array([valor_original]),
                metric="woe",
            )
        except Exception as error:
            raise ValueError(
                f"No se pudo transformar '{nombre_original}' a WoE: "
                f"{error}"
            ) from error

        datos_woe[nombre_variable_modelo] = float(
            np.asarray(valor_transformado).reshape(-1)[0]
        )

    return pd.DataFrame(
        [datos_woe],
        columns=variables,
    )


# ============================================================
# 8. PREPARAR LOS DATOS PARA EL MODELO
# ============================================================

def preparar_entrada(cliente):
    datos = obtener_datos_cliente(cliente)
    variables = obtener_variables_modelo()

    if not variables:
        raise ValueError(
            "No se pudieron determinar las variables del modelo."
        )

    if modelo_usa_woe():
        entrada = transformar_woe(datos, variables)

    else:
        faltantes = [
            variable
            for variable in variables
            if variable not in datos
        ]

        if faltantes:
            raise ValueError(
                f"Faltan variables de entrada: {faltantes}"
            )

        entrada = pd.DataFrame(
            [
                {
                    variable: datos[variable]
                    for variable in variables
                }
            ],
            columns=variables,
        )

    entrada = entrada[variables]

    if entrada.isnull().any().any():
        raise ValueError(
            "La entrada contiene valores nulos después de prepararla."
        )

    if not np.isfinite(entrada.to_numpy(dtype=float)).all():
        raise ValueError(
            "La entrada contiene valores no finitos."
        )

    return entrada, variables


# ============================================================
# 9. OBTENER PROBABILIDAD DE DEFAULT
# ============================================================

def obtener_probabilidad_default(entrada):
    probabilidades = np.asarray(modelo.predict_proba(entrada))

    if probabilidades.ndim != 2 or probabilidades.shape[0] < 1:
        raise ValueError(
            "El modelo devolvió una matriz de probabilidades inválida."
        )

    clases = getattr(modelo, "classes_", None)

    # Identificar la columna que corresponde a la clase positiva.
    # En este proyecto, el target esperado es 1 = default.
    if clases is not None:
        clases_lista = list(clases)

        if 1 in clases_lista:
            indice_default = clases_lista.index(1)
        elif "DEFAULT" in clases_lista:
            indice_default = clases_lista.index("DEFAULT")
        else:
            indice_default = 1

    else:
        indice_default = 1

    if indice_default >= probabilidades.shape[1]:
        raise ValueError(
            "No se encontró la columna de probabilidad de default. "
            f"Clases disponibles: {clases}"
        )

    probabilidad = float(probabilidades[0, indice_default])

    if not np.isfinite(probabilidad):
        raise ValueError(
            "El modelo devolvió una probabilidad no válida."
        )

    if probabilidad < 0 or probabilidad > 1:
        raise ValueError(
            "La probabilidad devuelta está fuera del rango [0, 1]."
        )

    return probabilidad


# ============================================================
# 10. ENDPOINT INICIAL
# ============================================================

@app.get("/")
def inicio():
    return {
        "mensaje": "Motor de Scoring Crediticio",
        "estado": "activo",
        "documentacion": "/docs",
        "modelo_cargado": modelo is not None,
    }


# ============================================================
# 11. ENDPOINT DE SALUD
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok" if modelo is not None else "error",
        "model_loaded": modelo is not None,
        "model_name": obtener_nombre_modelo(),
        "scenario": obtener_escenario(),
        "error": error_carga,
    }


# ============================================================
# 12. INFORMACIÓN DEL MODELO
# ============================================================

@app.get("/model-info")
def model_info():
    variables = obtener_variables_modelo()

    return {
        "model_loaded": modelo is not None,
        "model_name": obtener_nombre_modelo(),
        "scenario": obtener_escenario(),
        "variables": variables,
        "feature_count": len(variables),
        "uses_woe": modelo_usa_woe(),
        "threshold": UMBRAL,
        "artifact": artefacto_cargado or ruta_modelo.name,
        "compatibility_mode": modo_compatibilidad,
        "error": error_carga,
        "risk_band_policy": {
            "muy_bajo": "< 5%",
            "moderado": "5% a menos de 12%",
            "alto": "12% a menos de 25%",
            "inaceptable": "25% o más",
            "nota": (
                "Bandas ilustrativas para el proyecto académico; "
                "no constituyen una política crediticia validada."
            ),
        },
    }


# ============================================================
# 13. ENDPOINT DE PREDICCIÓN
# ============================================================

@app.post("/predict")
def predecir_riesgo(cliente: Prospecto):
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No hay un modelo cargado. "
                f"Detalle: {error_carga or 'Error desconocido'}"
            ),
        )

    try:
        datos = obtener_datos_cliente(cliente)

        # Validación lógica de la antigüedad laboral.
        antiguedad_maxima = max(
            0,
            (datos["edad_titular"] - 18) * 12,
        )

        if datos["antiguedad_laboral_meses"] > antiguedad_maxima:
            raise ValueError(
                "La antigüedad laboral no puede superar los meses "
                "posibles transcurridos desde los 18 años."
            )

        entrada, variables = preparar_entrada(cliente)

        probabilidad = obtener_probabilidad_default(entrada)
        probabilidad_porcentaje = probabilidad * 100

        # Clasificación estadística según el umbral del modelo.
        clasificacion = (
            "DEFAULT"
            if probabilidad >= UMBRAL
            else "NO DEFAULT"
        )

        # Bandas ilustrativas del proyecto.
        if probabilidad_porcentaje < 5:
            banda = "A - Riesgo Muy Bajo"
            decision = "APROBADO AUTOMÁTICO"

        elif probabilidad_porcentaje < 12:
            banda = "B - Riesgo Moderado"
            decision = "APROBADO CON REVISIÓN"

        elif probabilidad_porcentaje < 25:
            banda = "C - Riesgo Alto"
            decision = "RECHAZADO"

        else:
            banda = "D - Riesgo Inaceptable"
            decision = "RECHAZADO AUTOMÁTICO"

        return {
            "probabilidad_default": round(
                probabilidad_porcentaje, 2
            ),
            "banda": banda,
            "decision": decision,
            "modelo": obtener_nombre_modelo(),
            "escenario": obtener_escenario(),
            "variables_utilizadas": variables,
            "usa_woe": modelo_usa_woe(),
            "umbral_modelo": round(UMBRAL * 100, 2),
            "clasificacion_umbral": clasificacion,
            "nota_politica": (
                "Las bandas de riesgo y las decisiones son reglas "
                "demostrativas del proyecto académico, no una "
                "política crediticia validada."
            ),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except Exception as error:
        print("\nERROR INTERNO EN /predict")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Error durante la predicción: {str(error)}",
        )


# ============================================================
# 14. INICIO DEL SERVIDOR
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


# Cargar los artefactos una vez definidas todas las funciones.
cargar_artefactos()
