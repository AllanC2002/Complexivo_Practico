from pathlib import Path

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

# ============================================================
# 1. CARGA Y CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RUTA_DATOS = BASE_DIR / "clientes_solicitud_credito_clean.csv"
TARGET = "estado_default"
UMBRAL = 0.5
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_SPLITS = 10

# Escenario de ocho variables: se utilizan todas las variables
# predictoras numericas consideradas en el proyecto.
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

ESCENARIOS = {
    "8_variables": VARIABLES_8,
}

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
# 2. VALIDACION Y PREPARACION DE LOS DATOS
# ============================================================

if not RUTA_DATOS.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo de datos: {RUTA_DATOS}\n"
        "Coloca clientes_solicitud_credito_clean.csv en la misma carpeta "
        "que este script o modifica RUTA_DATOS."
    )

df = pd.read_csv(RUTA_DATOS)

VARIABLES_NECESARIAS = sorted(set(VARIABLES_8 + [TARGET]))
columnas_faltantes = [
    columna for columna in VARIABLES_NECESARIAS
    if columna not in df.columns
]

if columnas_faltantes:
    raise ValueError(
        f"Faltan columnas requeridas en el CSV: {columnas_faltantes}"
    )

# Eliminar unicamente filas cuyo target este vacio.
df = df.dropna(subset=[TARGET]).copy()

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("El target debe contener solamente 0 y 1.")

# Asegurar que las variables predictoras sean numericas.
for variable in VARIABLES_8:
    df[variable] = pd.to_numeric(df[variable], errors="coerce")

if df[VARIABLES_8].isnull().any().any():
    columnas_con_nulos = df[VARIABLES_8].columns[
        df[VARIABLES_8].isnull().any()
    ].tolist()
    raise ValueError(
        "Existen valores nulos o no numericos en las variables predictoras: "
        f"{columnas_con_nulos}. Revisa el proceso de limpieza antes de entrenar."
    )

if set(df[TARGET].unique()) != {0, 1}:
    raise ValueError(
        "El dataset debe contener las dos clases del target: 0 y 1."
    )

X = df[VARIABLES_8].copy()
y = df[TARGET].astype(int)

# Se mantiene una sola particion para que la comparacion sea reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("=" * 70)
print("EVALUACION DEL ESCENARIO DE OCHO VARIABLES")
print("=" * 70)
print("Registros utilizados:", len(df))
print("Entrenamiento:", len(X_train))
print("Prueba independiente:", len(X_test))
print(f"Tasa de default total: {y.mean() * 100:.2f}%")
print("Validacion cruzada:", N_SPLITS, "folds estratificados")
print("Variables utilizadas:", VARIABLES_8)


# ============================================================
# 3. BINNING OPTIMO Y TRANSFORMACION WoE
# ============================================================

def transformar_woe(X_ajuste, y_ajuste, X_aplicar, variables):
    """Ajusta los bins con los datos de ajuste y los aplica sin reajustar."""
    ajuste_woe = pd.DataFrame(index=X_ajuste.index)
    aplicar_woe = pd.DataFrame(index=X_aplicar.index)

    for variable in variables:
        binning = OptimalBinning(
            name=variable,
            dtype="numerical",
            solver="cp",
            max_n_prebins=20,
        )

        # Los bins se aprenden exclusivamente con el subconjunto de ajuste.
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
# 4. METRICAS
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
        "Precision": precision_score(
            y_real, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_real, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_real, y_pred, zero_division=0
        ),
        "Brier": brier_score_loss(y_real, probabilidades),
    }


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
# 6. VALIDACION CRUZADA ESTRATIFICADA DE 10 FOLDS
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

filas_cv_completas = []
resultados_por_escenario = {}

for nombre_escenario, variables in ESCENARIOS.items():
    print("\n" + "=" * 70)
    print("VALIDACION CRUZADA:", nombre_escenario)
    print("Variables:", variables)
    print("=" * 70)

    resultados_cv = {nombre: [] for nombre in NOMBRES_MODELOS}

    for numero_fold, (idx_train, idx_val) in enumerate(
        cv.split(X_train, y_train), start=1
    ):
        X_ent = X_train.iloc[idx_train][variables]
        X_val = X_train.iloc[idx_val][variables]
        y_ent = y_train.iloc[idx_train]
        y_val = y_train.iloc[idx_val]

        # WoE se ajusta dentro del fold para evitar fuga de informacion.
        X_ent_woe, X_val_woe = transformar_woe(
            X_ent, y_ent, X_val, variables
        )

        print(f"Fold {numero_fold}/{N_SPLITS} completado")

        for nombre_modelo in NOMBRES_MODELOS:
            modelo_cv = crear_modelo(nombre_modelo)

            if es_modelo_logistico(nombre_modelo):
                modelo_cv.fit(X_ent_woe, y_ent)
                probabilidades = modelo_cv.predict_proba(X_val_woe)[:, 1]
            else:
                # Random Forest y XGBoost reciben las variables originales.
                modelo_cv.fit(X_ent, y_ent)
                probabilidades = modelo_cv.predict_proba(X_val)[:, 1]

            metricas = calcular_metricas(y_val, probabilidades)
            resultados_cv[nombre_modelo].append(metricas)

    filas_escenario = []

    for nombre_modelo, resultados in resultados_cv.items():
        fila = {
            "Escenario": nombre_escenario,
            "Modelo": nombre_modelo,
            "Numero_variables": len(variables),
            "Variables": ", ".join(variables),
        }

        for metrica in METRICAS:
            valores = [resultado[metrica] for resultado in resultados]
            fila[f"{metrica}_media"] = float(np.mean(valores))
            fila[f"{metrica}_desv"] = float(np.std(valores, ddof=1))

        filas_escenario.append(fila)

    tabla_escenario = pd.DataFrame(filas_escenario).sort_values(
        "AUC_media", ascending=False
    )
    resultados_por_escenario[nombre_escenario] = tabla_escenario
    filas_cv_completas.extend(filas_escenario)

    print("\nResultados de validacion cruzada:")
    print(
        tabla_escenario[
            ["Modelo", "AUC_media", "Gini_media", "KS_media", "Brier_media"]
        ].round(4).to_string(index=False)
    )


# Guardar comparacion de modelos para el escenario de ocho variables.
tabla_cv_completa = pd.DataFrame(filas_cv_completas).sort_values(
    ["Escenario", "AUC_media"], ascending=[True, False]
)
tabla_cv_completa.to_csv(
    BASE_DIR / "comparacion_modelos_8_variables.csv",
    index=False,
)

print("\nArchivo de comparacion guardado: comparacion_modelos_8_variables.csv")


# ============================================================
# 7. SELECCION DEL MODELO DENTRO DEL ESCENARIO
# ============================================================

def seleccionar_modelo(tabla_cv):
    validos = tabla_cv[
        (tabla_cv["Gini_media"] > 0.60)
        & (tabla_cv["KS_media"] > 40)
    ]

    if validos.empty:
        print("Ningun modelo cumple ambos umbrales promedio.")
        return tabla_cv.sort_values(
            "AUC_media", ascending=False
        ).iloc[0]["Modelo"]

    mejor_auc = validos["AUC_media"].max()

    # Si un modelo logistico esta a no mas de 0.02 de AUC del mejor,
    # se prioriza la interpretabilidad de la regresion logistica.
    logisticos = validos[
        validos["Modelo"].str.startswith("Regresion Logistica")
        & (validos["AUC_media"] >= mejor_auc - 0.02)
    ]

    if not logisticos.empty:
        return logisticos.sort_values(
            "AUC_media", ascending=False
        ).iloc[0]["Modelo"]

    return validos.sort_values(
        "AUC_media", ascending=False
    ).iloc[0]["Modelo"]


# ============================================================
# 8. ENTRENAMIENTO FINAL Y EVALUACION EN TEST
# ============================================================

filas_test = []
filas_matriz = []

for nombre_escenario, variables in ESCENARIOS.items():
    tabla_cv = resultados_por_escenario[nombre_escenario]
    modelo_elegido = seleccionar_modelo(tabla_cv)

    print("\n" + "=" * 70)
    print("EVALUACION FINAL EN TEST:", nombre_escenario)
    print("Modelo seleccionado por CV:", modelo_elegido)
    print("Variables:", variables)
    print("=" * 70)

    X_train_final = X_train[variables]
    X_test_final = X_test[variables]
    modelo_final = crear_modelo(modelo_elegido)

    if es_modelo_logistico(modelo_elegido):
        X_train_woe, X_test_woe = transformar_woe(
            X_train_final,
            y_train,
            X_test_final,
            variables,
        )
        modelo_final.fit(X_train_woe, y_train)
        probabilidades_test = modelo_final.predict_proba(X_test_woe)[:, 1]
    else:
        modelo_final.fit(X_train_final, y_train)
        probabilidades_test = modelo_final.predict_proba(X_test_final)[:, 1]

    metricas_test = calcular_metricas(y_test, probabilidades_test)

    filas_test.append({
        "Escenario": nombre_escenario,
        "Modelo": modelo_elegido,
        "Numero_variables": len(variables),
        "Variables": ", ".join(variables),
        **metricas_test,
    })

    y_pred_test = (probabilidades_test >= UMBRAL).astype(int)
    matriz = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()

    filas_matriz.append({
        "Escenario": nombre_escenario,
        "Modelo": modelo_elegido,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    })

    print("\nMetricas de test:")
    for metrica in METRICAS:
        print(f"{metrica}: {metricas_test[metrica]:.4f}")

    print("\nMatriz de confusion:")
    print(pd.DataFrame(
        matriz,
        index=["Real: no default (0)", "Real: default (1)"],
        columns=["Predicho: 0", "Predicho: 1"],
    ).to_string())

    print("\nVerificacion de umbrales en test:")
    print(
        "Gini > 0.60:",
        "CUMPLE" if metricas_test["Gini"] > 0.60 else "NO CUMPLE",
    )
    print(
        "KS > 40:",
        "CUMPLE" if metricas_test["KS"] > 40 else "NO CUMPLE",
    )


# ============================================================
# 9. GUARDAR RESULTADOS FINALES
# ============================================================

tabla_test = pd.DataFrame(filas_test)
tabla_test.to_csv(
    BASE_DIR / "resultados_test_8_variables.csv",
    index=False,
)

tabla_matrices = pd.DataFrame(filas_matriz)
tabla_matrices.to_csv(
    BASE_DIR / "matrices_confusion_8_variables.csv",
    index=False,
)

print("\n" + "=" * 70)
print("RESULTADO FINAL DEL ESCENARIO DE OCHO VARIABLES EN TEST")
print("=" * 70)
print(
    tabla_test[
        ["Escenario", "Modelo", "AUC", "Gini", "KS", "Brier", "Recall", "Precision"]
    ].round(4).to_string(index=False)
)

print("\nArchivos generados:")
print("- comparacion_modelos_8_variables.csv")
print("- resultados_test_8_variables.csv")
print("- matrices_confusion_8_variables.csv")
print("\nProceso terminado.")
