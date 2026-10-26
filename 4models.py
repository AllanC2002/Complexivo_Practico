from pathlib import Path

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RUTA_DATOS = BASE_DIR / "clientes_solicitud_credito_clean.csv"
TARGET = "estado_default"
UMBRAL = 0.5
TEST_SIZE = 0.20
RANDOM_STATE = 42

VARIABLES_CANDIDATAS = [
    "edad_titular",
    "ingreso_mensual_usd",
    "ratio_deuda_ingreso",
    "antiguedad_laboral_meses",
    "consultas_buro_ultimos_6m",
    "meses_maximo_mora_historica",
    "score_buro_externo",
    "monto_credito_solicitado",
]

# Escenario 6A: incluye monto solicitado y excluye ingreso mensual.
FEATURES_6A = [
    "ratio_deuda_ingreso",
    "meses_maximo_mora_historica",
    "score_buro_externo",
    "antiguedad_laboral_meses",
    "consultas_buro_ultimos_6m",
    "monto_credito_solicitado",
]

# Escenario 6B: incluye ingreso mensual y excluye monto solicitado.
FEATURES_6B = [
    "ratio_deuda_ingreso",
    "meses_maximo_mora_historica",
    "score_buro_externo",
    "antiguedad_laboral_meses",
    "consultas_buro_ultimos_6m",
    "ingreso_mensual_usd",
]

NOMBRES_MODELOS = [
    "Regresion Logistica L1",
    "Regresion Logistica L2",
    "Random Forest",
    "XGBoost",
]

METRICAS = [
    "AUC",
    "Gini",
    "KS",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Brier",
]


# ============================================================
# 2. CARGA, VALIDACION Y PARTICION DE LOS DATOS
# ============================================================

if not RUTA_DATOS.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo: {RUTA_DATOS}\n"
        "Coloca clientes_solicitud_credito_clean.csv en la misma carpeta "
        "que este script."
    )

df = pd.read_csv(RUTA_DATOS)

columnas_necesarias = VARIABLES_CANDIDATAS + [TARGET]
columnas_faltantes = [
    columna for columna in columnas_necesarias
    if columna not in df.columns
]

if columnas_faltantes:
    raise ValueError(
        f"Faltan columnas requeridas en el CSV: {columnas_faltantes}"
    )

df = df.dropna(subset=[TARGET]).copy()

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("El target debe contener solamente 0 y 1.")

if df[TARGET].nunique() != 2:
    raise ValueError("El conjunto debe contener las dos clases del target.")

for variable in VARIABLES_CANDIDATAS:
    df[variable] = pd.to_numeric(df[variable], errors="coerce")

if df[VARIABLES_CANDIDATAS].isnull().any().any():
    columnas_con_nulos = df[VARIABLES_CANDIDATAS].columns[
        df[VARIABLES_CANDIDATAS].isnull().any()
    ].tolist()
    raise ValueError(
        "Existen nulos o valores no numericos en los predictores: "
        f"{columnas_con_nulos}. Revisa el archivo limpio."
    )

X = df[VARIABLES_CANDIDATAS].copy()
y = df[TARGET].astype(int)

# Misma configuracion de particion usada por model5v.py, model6v.py y model8v.py.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("=" * 72)
print("EVALUACION EN TEST DE TODOS LOS MODELOS Y ESCENARIOS")
print("=" * 72)
print("Registros utilizados:", len(df))
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))
print(f"Tasa de default: {y.mean():.2%}")
print("Umbral de clasificacion:", UMBRAL)


# ============================================================
# 3. SELECCION POR IV PARA EL ESCENARIO DE CINCO VARIABLES
# ============================================================

def calcular_iv(variable, y_entrenamiento):
    if variable.nunique() > 10:
        grupos = pd.qcut(
            variable,
            q=10,
            duplicates="drop",
        )
    else:
        grupos = variable.astype(str)

    tabla = pd.crosstab(grupos, y_entrenamiento)
    tabla = tabla.reindex(columns=[0, 1], fill_value=0).astype(float)

    # Suavizado para evitar logaritmos de cero.
    tabla += 0.5

    dist_buenos = tabla[0] / tabla[0].sum()
    dist_malos = tabla[1] / tabla[1].sum()
    woe = np.log(dist_buenos / dist_malos)

    return float(((dist_buenos - dist_malos) * woe).sum())


def seleccionar_top_iv(X_entrenamiento, y_entrenamiento, variables_candidatas):
    puntuaciones = {}

    for variable in variables_candidatas:
        puntuaciones[variable] = calcular_iv(
            X_entrenamiento[variable],
            y_entrenamiento,
        )

    ordenadas = sorted(
        puntuaciones,
        key=puntuaciones.get,
        reverse=True,
    )

    return ordenadas[:5]


# Selecciona las cinco variables usando solo la particion de entrenamiento.
VARIABLES_5 = seleccionar_top_iv(
    X_train,
    y_train,
    VARIABLES_CANDIDATAS,
)

ESCENARIOS = {
    "5_variables": VARIABLES_5,
    "6A_sin_ingreso_con_monto": FEATURES_6A,
    "6B_con_ingreso_sin_monto": FEATURES_6B,
    "8_variables": VARIABLES_CANDIDATAS.copy(),
}

print("\nVariables elegidas para el escenario de cinco variables:")
for variable in VARIABLES_5:
    print("-", variable)


# ============================================================
# 4. TRANSFORMACION WoE PARA REGRESION LOGISTICA
# ============================================================

def transformar_woe(X_ajuste, y_ajuste, X_aplicar, variables):
    """Ajusta WoE con entrenamiento y aplica los mismos bins a test."""
    ajuste_woe = pd.DataFrame(index=X_ajuste.index)
    aplicar_woe = pd.DataFrame(index=X_aplicar.index)

    for variable in variables:
        binning = OptimalBinning(
            name=variable,
            dtype="numerical",
            solver="cp",
            max_n_prebins=20,
        )

        binning.fit(
            X_ajuste[variable].to_numpy(),
            y_ajuste.to_numpy(),
        )

        ajuste_woe[variable] = binning.transform(
            X_ajuste[variable].to_numpy(),
            metric="woe",
        )

        aplicar_woe[variable] = binning.transform(
            X_aplicar[variable].to_numpy(),
            metric="woe",
        )

    return ajuste_woe, aplicar_woe


# ============================================================
# 5. DEFINICION DE LOS MODELOS
# ============================================================

def crear_modelo(nombre):
    if nombre == "Regresion Logistica L1":
        return Pipeline([
            ("escalador", StandardScaler()),
            ("modelo", LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1.0,
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ])

    if nombre == "Regresion Logistica L2":
        return Pipeline([
            ("escalador", StandardScaler()),
            ("modelo", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                C=1.0,
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ])

    if nombre == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if nombre == "XGBoost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Modelo no reconocido: {nombre}")


def es_modelo_logistico(nombre):
    return nombre.startswith("Regresion Logistica")


# ============================================================
# 6. METRICAS DE TEST
# ============================================================

def calcular_metricas(y_real, probabilidades):
    probabilidades = np.asarray(probabilidades)
    y_pred = (probabilidades >= UMBRAL).astype(int)

    auc = roc_auc_score(y_real, probabilidades)
    fpr, tpr, _ = roc_curve(y_real, probabilidades)
    ks = np.max(tpr - fpr) * 100

    return {
        "AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": ks,
        "Accuracy": accuracy_score(y_real, y_pred),
        "Precision": precision_score(y_real, y_pred, zero_division=0),
        "Recall": recall_score(y_real, y_pred, zero_division=0),
        "F1": f1_score(y_real, y_pred, zero_division=0),
        "Brier": brier_score_loss(y_real, probabilidades),
    }


# ============================================================
# 7. ENTRENAR CADA MODELO Y EVALUARLO EN EL CONJUNTO DE PRUEBA
# ============================================================

filas_test = []
filas_matrices = []

for nombre_escenario, variables in ESCENARIOS.items():
    print("\n" + "=" * 72)
    print("ESCENARIO:", nombre_escenario)
    print("Numero de variables:", len(variables))
    print("Variables:", ", ".join(variables))
    print("=" * 72)

    X_train_escenario = X_train[variables].copy()
    X_test_escenario = X_test[variables].copy()

    for nombre_modelo in NOMBRES_MODELOS:
        print(f"Evaluando {nombre_modelo}...")
        modelo = crear_modelo(nombre_modelo)

        if es_modelo_logistico(nombre_modelo):
            X_train_modelo, X_test_modelo = transformar_woe(
                X_train_escenario,
                y_train,
                X_test_escenario,
                variables,
            )
        else:
            # Los modelos basados en arboles usan las variables originales.
            X_train_modelo = X_train_escenario
            X_test_modelo = X_test_escenario

        modelo.fit(X_train_modelo, y_train)
        probabilidades_test = modelo.predict_proba(X_test_modelo)[:, 1]
        metricas_test = calcular_metricas(y_test, probabilidades_test)

        # Guardar metricas de prueba con esquema comparable entre escenarios.
        filas_test.append({
            "Escenario": nombre_escenario,
            "Modelo": nombre_modelo,
            "Numero_variables": len(variables),
            "Variables": ", ".join(variables),
            **metricas_test,
        })

        # Guardar matriz de confusion para este modelo y escenario.
        y_pred_test = (probabilidades_test >= UMBRAL).astype(int)
        matriz = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
        tn, fp, fn, tp = matriz.ravel()

        filas_matrices.append({
            "Escenario": nombre_escenario,
            "Modelo": nombre_modelo,
            "Numero_variables": len(variables),
            "Variables": ", ".join(variables),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

        print(
            f"  AUC={metricas_test['AUC']:.4f} | "
            f"Gini={metricas_test['Gini']:.4f} | "
            f"KS={metricas_test['KS']:.2f} | "
            f"Brier={metricas_test['Brier']:.4f}"
        )
        print(f"  Matriz: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# ============================================================
# 8. EXPORTAR LOS DOS CSV CONSOLIDADOS
# ============================================================

tabla_test = pd.DataFrame(filas_test)
tabla_test = tabla_test.sort_values(
    ["Escenario", "AUC"],
    ascending=[True, False],
)
tabla_test.to_csv(
    BASE_DIR / "resultados_test_todos_modelos.csv",
    index=False,
)

tabla_matrices = pd.DataFrame(filas_matrices)
tabla_matrices = tabla_matrices.sort_values(
    ["Escenario", "Modelo"]
)
tabla_matrices.to_csv(
    BASE_DIR / "matrices_confusion_todos_modelos.csv",
    index=False,
)

print("\n" + "=" * 72)
print("RESUMEN DE TEST: TODOS LOS MODELOS")
print("=" * 72)
print(
    tabla_test[
        ["Escenario", "Modelo", "AUC", "Gini", "KS", "Accuracy", "Precision", "Recall", "F1", "Brier"]
    ].round(4).to_string(index=False)
)

print("\nArchivos generados:")
print("- resultados_test_todos_modelos.csv")
print("- matrices_confusion_todos_modelos.csv")
print("\nTotal de evaluaciones de modelos:", len(tabla_test))
print("Total de matrices de confusion:", len(tabla_matrices))
print("Proceso terminado.")
