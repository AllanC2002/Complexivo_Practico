import pandas as pd
import numpy as np
import joblib
import json

from optbinning import OptimalBinning

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# 1. CONFIGURACION DEL MODELO DEFINITIVO

# Opciones: "5" u "8"
ESCENARIO = "8"

# Opciones:
# "Regresion Logistica L1"
# "Regresion Logistica L2"
# "Random Forest"
# "XGBoost"

MODELO_FINAL = "XGBoost"

UMBRAL = 0.5
TARGET = "estado_default"

VARIABLES_CANDIDATAS = [
    'edad_titular',
    'ingreso_mensual_usd',
    'ratio_deuda_ingreso',
    'antiguedad_laboral_meses',
    'consultas_buro_ultimos_6m',
    'meses_maximo_mora_historica',
    'score_buro_externo',
    'monto_credito_solicitado'
]


# 2. CARGAR EL DATASET LIMPIO

df = pd.read_csv('data/processed/clientes_solicitud_credito_clean.csv')

df = df.dropna(subset=[TARGET]).copy()

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("El target debe contener solo 0 y 1.")

if df[TARGET].nunique() != 2:
    raise ValueError("Se necesitan las dos clases del target.")

if df[VARIABLES_CANDIDATAS].isnull().any().any():
    raise ValueError(
        "Existen nulos en las variables predictoras. "
        "Revisa primero la limpieza."
    )

X = df[VARIABLES_CANDIDATAS].copy()
y = df[TARGET].astype(int)

print("Registros utilizados para el entrenamiento final:", len(df))
print("Tasa de default:", f"{y.mean():.2%}")


# 3. SELECCION FINAL DE VARIABLES

def calcular_iv(variable, target):

    if variable.nunique() > 10:
        grupos = pd.qcut(
            variable,
            q=10,
            duplicates='drop'
        )
    else:
        grupos = variable.astype('string').fillna('NULO')

    tabla = pd.crosstab(grupos, target)

    tabla = tabla.reindex(
        columns=[0, 1],
        fill_value=0
    ).astype(float)

    # Suavizado para evitar logaritmos de cero.
    tabla += 0.5

    dist_buenos = tabla[0] / tabla[0].sum()
    dist_malos = tabla[1] / tabla[1].sum()

    woe = np.log(dist_buenos / dist_malos)

    return float(
        ((dist_buenos - dist_malos) * woe).sum()
    )


if ESCENARIO == "5":

    puntuaciones_iv = {
        variable: calcular_iv(X[variable], y)
        for variable in VARIABLES_CANDIDATAS
    }

    tabla_iv = pd.DataFrame([
        {
            'Variable': variable,
            'IV': valor
        }
        for variable, valor in puntuaciones_iv.items()
    ]).sort_values('IV', ascending=False)

    VARIABLES_FINALES = tabla_iv.head(5)['Variable'].tolist()

    tabla_iv.to_csv(
        'data/reference/iv_seleccion_final.csv',
        index=False
    )

elif ESCENARIO == "8":

    VARIABLES_FINALES = VARIABLES_CANDIDATAS.copy()

else:
    raise ValueError("ESCENARIO debe ser '5' u '8'.")


X_final = X[VARIABLES_FINALES].copy()

print("\nVariables definitivas:")
for variable in VARIABLES_FINALES:
    print("-", variable)

print("Cantidad de variables:", len(VARIABLES_FINALES))

# 4. CREAR EL ALGORITMO SELECCIONADO

def crear_modelo(nombre):

    if nombre == "Regresion Logistica L1":
        return Pipeline([
            ('escalador', StandardScaler()),
            ('modelo', LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=1.0,
                max_iter=2000,
                random_state=42
            ))
        ])

    elif nombre == "Regresion Logistica L2":
        return Pipeline([
            ('escalador', StandardScaler()),
            ('modelo', LogisticRegression(
                penalty='l2',
                solver='liblinear',
                C=1.0,
                max_iter=2000,
                random_state=42
            ))
        ])

    elif nombre == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )

    elif nombre == "XGBoost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError(
            f"Modelo no reconocido: {nombre}"
        )


modelo = crear_modelo(MODELO_FINAL)

USA_WOE = MODELO_FINAL.startswith('Regresion Logistica')

# 5. BINNING OPTIMO Y TRANSFORMACION WoE

binnings = {}
tablas_binning = []

if USA_WOE:

    X_entrenamiento = pd.DataFrame(index=X_final.index)

    for variable in VARIABLES_FINALES:

        optb = OptimalBinning(
            name=variable,
            dtype='numerical',
            solver='cp',
            max_n_prebins=20
        )

        # En el modelo definitivo se ajusta con toda la data.
        optb.fit(
            X_final[variable].to_numpy(),
            y.to_numpy()
        )

        X_entrenamiento[variable] = optb.transform(
            X_final[variable].to_numpy(),
            metric='woe'
        )

        binnings[variable] = optb

        # Guardar tabla de intervalos, distribución y WoE.
        tabla = optb.binning_table.build().copy()
        tabla.insert(0, 'Variable', variable)
        tablas_binning.append(tabla)

    X_modelo = X_entrenamiento

else:

    # Los modelos de árboles usan las variables originales.
    X_modelo = X_final.copy()

# 6. ENTRENAR CON TODA LA DATA ETIQUETADA

modelo.fit(X_modelo, y)

print("\nModelo entrenado correctamente.")
print("Algoritmo:", MODELO_FINAL)
print("Registros utilizados:", len(X_modelo))
print("Transformacion WoE aplicada:", USA_WOE)

# 7. EXPORTAR MODELO Y COMPONENTES

artefacto = {
    'modelo': modelo,
    'variables': VARIABLES_FINALES,
    'escenario': ESCENARIO,
    'nombre_modelo': MODELO_FINAL,
    'usa_woe': USA_WOE,
    'binnings': binnings,
    'umbral': UMBRAL,
    'target': TARGET,
    'registros_entrenamiento': len(X_modelo)
}

nombre_archivo = (
    f"modelo_scoring_{ESCENARIO}_variables_"
    f"{MODELO_FINAL.lower().replace(' ', '_')}.joblib"
)

joblib.dump(
    artefacto,
    f"models/{nombre_archivo}",
    compress=3
)

# Guardar las variables seleccionadas para auditoría.
pd.DataFrame({
    'Variable': VARIABLES_FINALES
}).to_csv(
    'data/reference/variables_modelo_final.csv',
    index=False
)

# Exportar el diccionario de binning y WoE si corresponde.
if USA_WOE and tablas_binning:

    diccionario_woe = pd.concat(
        tablas_binning,
        ignore_index=True
    )

    diccionario_woe.to_csv(
        'data/reference/diccionario_binning_woe.csv',
        index=False
    )

# Guardar un resumen legible de la configuración.
metadata = {
    'escenario': ESCENARIO,
    'modelo': MODELO_FINAL,
    'variables': VARIABLES_FINALES,
    'usa_woe': USA_WOE,
    'umbral': UMBRAL,
    'registros_entrenamiento': len(X_modelo),
    'tasa_default_entrenamiento': float(y.mean())
}

with open(
    'models/metadata_modelo_final.json',
    'w',
    encoding='utf-8'
) as archivo:
    json.dump(
        metadata,
        archivo,
        ensure_ascii=False,
        indent=4
    )

print("\nEXPORTACION FINALIZADA")
print("Modelo:", nombre_archivo)
print("Variables: variables_modelo_final.csv")
print("Configuracion: metadata_modelo_final.json")

if USA_WOE:
    print("Diccionario WoE: diccionario_binning_woe.csv")