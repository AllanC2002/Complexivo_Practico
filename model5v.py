from pathlib import Path

import pandas as pd
import numpy as np
from optbinning import OptimalBinning

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss
)

# ============================================================
# 1. CARGA Y CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RUTA_DATOS = BASE_DIR / 'clientes_solicitud_credito_clean.csv'

if not RUTA_DATOS.exists():
    raise FileNotFoundError(
        f'No se encontro el archivo de datos: {RUTA_DATOS}. '
        'Coloca clientes_solicitud_credito_clean.csv en la misma carpeta que este script.'
    )

df = pd.read_csv(RUTA_DATOS)

TARGET = 'estado_default'
UMBRAL = 0.5
NUM_VARIABLES = 5

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

df = df.dropna(subset=[TARGET]).copy()

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("El target debe contener solo 0 y 1.")

if df[VARIABLES_CANDIDATAS].isnull().any().any():
    raise ValueError("Existen nulos en las variables predictoras.")

X = df[VARIABLES_CANDIDATAS].copy()
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Registros totales:", len(df))
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))


# ============================================================
# 2. CALCULO DE IV Y SELECCION DE VARIABLES
# ============================================================

def calcular_iv(variable, y_entrenamiento):

    if variable.nunique() > 10:
        grupos = pd.qcut(
            variable,
            q=10,
            duplicates='drop'
        )
    else:
        grupos = variable.astype(str)

    tabla = pd.crosstab(grupos, y_entrenamiento)

    tabla = tabla.reindex(
        columns=[0, 1],
        fill_value=0
    ).astype(float)

    # Suavizado para evitar logaritmos de cero.
    tabla += 0.5

    dist_buenos = tabla[0] / tabla[0].sum()
    dist_malos = tabla[1] / tabla[1].sum()

    woe = np.log(dist_buenos / dist_malos)

    iv = ((dist_buenos - dist_malos) * woe).sum()

    return float(iv)


def seleccionar_top_iv(X_entrenamiento, y_entrenamiento):

    puntuaciones = {}

    for variable in VARIABLES_CANDIDATAS:
        puntuaciones[variable] = calcular_iv(
            X_entrenamiento[variable],
            y_entrenamiento
        )

    ordenadas = sorted(
        puntuaciones,
        key=puntuaciones.get,
        reverse=True
    )

    return ordenadas[:NUM_VARIABLES]


# ============================================================
# 3. BINNING OPTIMO Y TRANSFORMACION WoE
# ============================================================

def transformar_woe(X_ajuste, y_ajuste, X_aplicar, variables):

    ajuste_woe = pd.DataFrame(index=X_ajuste.index)
    aplicar_woe = pd.DataFrame(index=X_aplicar.index)

    for variable in variables:

        binning = OptimalBinning(
            name=variable,
            dtype='numerical',
            solver='cp',
            max_n_prebins=20
        )

        # Aprende los intervalos y WoE con los datos de ajuste.
        binning.fit(
            X_ajuste[variable].to_numpy(),
            y_ajuste.to_numpy()
        )

        ajuste_woe[variable] = binning.transform(
            X_ajuste[variable].to_numpy(),
            metric='woe'
        )

        # Aplica la misma transformacion a los datos nuevos.
        aplicar_woe[variable] = binning.transform(
            X_aplicar[variable].to_numpy(),
            metric='woe'
        )

    return ajuste_woe, aplicar_woe


# ============================================================
# 4. METRICAS
# ============================================================

def calcular_metricas(y_real, probabilidades):

    y_pred = (
        np.asarray(probabilidades) >= UMBRAL
    ).astype(int)

    auc = roc_auc_score(y_real, probabilidades)

    fpr, tpr, _ = roc_curve(
        y_real,
        probabilidades
    )

    ks = np.max(tpr - fpr) * 100

    return {
        'AUC': auc,
        'Gini': 2 * auc - 1,
        'KS': ks,
        'Accuracy': accuracy_score(y_real, y_pred),
        'Precision': precision_score(
            y_real,
            y_pred,
            zero_division=0
        ),
        'Recall': recall_score(
            y_real,
            y_pred,
            zero_division=0
        ),
        'F1': f1_score(
            y_real,
            y_pred,
            zero_division=0
        ),
        'Brier': brier_score_loss(
            y_real,
            probabilidades
        )
    }


# ============================================================
# 5. MODELOS
# ============================================================

def crear_modelo(nombre):

    if nombre == 'Regresion Logistica L1':
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

    if nombre == 'Regresion Logistica L2':
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

    if nombre == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )

    if nombre == 'XGBoost':
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

    raise ValueError("Modelo no reconocido.")


NOMBRES_MODELOS = [
    'Regresion Logistica L1',
    'Regresion Logistica L2',
    'Random Forest',
    'XGBoost'
]

METRICAS = [
    'AUC',
    'Gini',
    'KS',
    'Accuracy',
    'Precision',
    'Recall',
    'F1',
    'Brier'
]


# ============================================================
# 6. VALIDACION CRUZADA ESTRATIFICADA DE 10 FOLDS
# ============================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

resultados_cv = {
    nombre: [] for nombre in NOMBRES_MODELOS
}

for numero_fold, (idx_train, idx_val) in enumerate(
    cv.split(X_train, y_train),
    start=1
):

    X_ent = X_train.iloc[idx_train]
    X_val = X_train.iloc[idx_val]

    y_ent = y_train.iloc[idx_train]
    y_val = y_train.iloc[idx_val]

    # La seleccion de variables usa solo el entrenamiento
    # de cada fold para evitar utilizar su validacion.
    variables_fold = seleccionar_top_iv(
        X_ent,
        y_ent
    )

    print(f"\nFold {numero_fold}/10")
    print("Variables seleccionadas:", variables_fold)

    X_ent_sel = X_ent[variables_fold]
    X_val_sel = X_val[variables_fold]

    # WoE se ajusta dentro del entrenamiento de cada fold.
    X_ent_woe, X_val_woe = transformar_woe(
        X_ent_sel,
        y_ent,
        X_val_sel,
        variables_fold
    )

    for nombre in NOMBRES_MODELOS:

        modelo = crear_modelo(nombre)

        if nombre.startswith('Regresion Logistica'):

            modelo.fit(
                X_ent_woe,
                y_ent
            )

            probabilidades = modelo.predict_proba(
                X_val_woe
            )[:, 1]

        else:

            modelo.fit(
                X_ent_sel,
                y_ent
            )

            probabilidades = modelo.predict_proba(
                X_val_sel
            )[:, 1]

        metricas = calcular_metricas(
            y_val,
            probabilidades
        )

        resultados_cv[nombre].append(metricas)


# ============================================================
# 7. RESUMEN DE VALIDACION CRUZADA
# ============================================================

filas = []

for nombre, resultados in resultados_cv.items():

    fila = {
        'Modelo': nombre
    }

    for metrica in METRICAS:

        valores = [
            resultado[metrica]
            for resultado in resultados
        ]

        fila[metrica + '_media'] = float(
            np.mean(valores)
        )

        fila[metrica + '_desv'] = float(
            np.std(valores, ddof=1)
        )

    filas.append(fila)

tabla_cv = pd.DataFrame(filas).sort_values(
    'AUC_media',
    ascending=False
)

print("\nRESULTADOS DE VALIDACION CRUZADA")
print(
    tabla_cv.round(4).to_string(index=False)
)


# ============================================================
# 8. SELECCION DEL MODELO
# ============================================================

validos = tabla_cv[
    (tabla_cv['Gini_media'] > 0.60)
    & (tabla_cv['KS_media'] > 40)
]

if validos.empty:

    print(
        "\nNingun modelo cumple ambos umbrales en promedio."
    )

    modelo_elegido = tabla_cv.iloc[0]['Modelo']

else:

    mejor_auc = validos['AUC_media'].max()

    # Favorecer regresion logistica si su AUC queda
    # a menos de 0.02 del mejor modelo valido.
    logisticos = validos[
        validos['Modelo'].str.startswith(
            'Regresion Logistica'
        )
        & (
            validos['AUC_media']
            >= mejor_auc - 0.02
        )
    ]

    if not logisticos.empty:

        modelo_elegido = logisticos.sort_values(
            'AUC_media',
            ascending=False
        ).iloc[0]['Modelo']

    else:

        modelo_elegido = validos.sort_values(
            'AUC_media',
            ascending=False
        ).iloc[0]['Modelo']

print("\nModelo seleccionado:", modelo_elegido)


# ============================================================
# 9. SELECCION FINAL Y ENTRENAMIENTO
# ============================================================

# Seleccionar las cinco variables usando solo entrenamiento.
# El conjunto de prueba no interviene en esta seleccion.
VARIABLES_FINALES = seleccionar_top_iv(
    X_train,
    y_train
)

print("\nCinco variables finales:")

for variable in VARIABLES_FINALES:
    print("-", variable)


# ============================================================
# 10. EXPORTACION ESTANDARIZADA DE COMPARACION
# ============================================================

# Cada modelo se identifica con el mismo escenario,
# cantidad de variables y nombres del conjunto final.
# Durante la validacion cruzada, las variables se seleccionaron
# nuevamente dentro de cada fold.
tabla_cv.insert(
    0,
    'Escenario',
    '5_variables'
)

tabla_cv.insert(
    2,
    'Numero_variables',
    len(VARIABLES_FINALES)
)

tabla_cv.insert(
    3,
    'Variables',
    ', '.join(VARIABLES_FINALES)
)

tabla_cv.to_csv(
    BASE_DIR / 'comparacion_modelos_5_variables.csv',
    index=False
)

print("\nComparacion de modelos estandarizada guardada:")
print("comparacion_modelos_5_variables.csv")


# ============================================================
# 11. ENTRENAMIENTO FINAL
# ============================================================

X_train_final = X_train[VARIABLES_FINALES]
X_test_final = X_test[VARIABLES_FINALES]

modelo_final = crear_modelo(
    modelo_elegido
)

if modelo_elegido.startswith('Regresion Logistica'):

    X_train_woe, X_test_woe = transformar_woe(
        X_train_final,
        y_train,
        X_test_final,
        VARIABLES_FINALES
    )

    modelo_final.fit(
        X_train_woe,
        y_train
    )

    probabilidades_test = modelo_final.predict_proba(
        X_test_woe
    )[:, 1]

else:

    modelo_final.fit(
        X_train_final,
        y_train
    )

    probabilidades_test = modelo_final.predict_proba(
        X_test_final
    )[:, 1]


# ============================================================
# 12. EVALUACION FINAL EN TEST
# ============================================================

metricas_test = calcular_metricas(
    y_test,
    probabilidades_test
)

print("\nRESULTADOS FINALES EN TEST")
print("Modelo:", modelo_elegido)
print("Umbral de clasificacion:", UMBRAL)

for metrica in METRICAS:
    print(
        f"{metrica}: {metricas_test[metrica]:.4f}"
    )


# ============================================================
# 13. MATRIZ DE CONFUSION
# ============================================================

y_pred_test = (
    probabilidades_test >= UMBRAL
).astype(int)

matriz = confusion_matrix(
    y_test,
    y_pred_test,
    labels=[0, 1]
)

print("\nMATRIZ DE CONFUSION")

print(
    pd.DataFrame(
        matriz,
        index=[
            'Real: no default (0)',
            'Real: default (1)'
        ],
        columns=[
            'Predicho: 0',
            'Predicho: 1'
        ]
    ).to_string()
)

tn, fp, fn, tp = matriz.ravel()


# ============================================================
# 14. EXPORTACION DE METRICAS DE TEST
# ============================================================

fila_test = {
    'Escenario': '5_variables',
    'Modelo': modelo_elegido,
    'Numero_variables': len(VARIABLES_FINALES),
    'Variables': ', '.join(VARIABLES_FINALES),
    **metricas_test
}

tabla_test = pd.DataFrame([
    fila_test
])

tabla_test.to_csv(
    BASE_DIR / 'resultados_test_5_variables.csv',
    index=False
)


# ============================================================
# 15. EXPORTACION DE MATRIZ DE CONFUSION
# ============================================================

tabla_matriz = pd.DataFrame([{
    'Escenario': '5_variables',
    'Modelo': modelo_elegido,
    'TN': int(tn),
    'FP': int(fp),
    'FN': int(fn),
    'TP': int(tp)
}])

tabla_matriz.to_csv(
    BASE_DIR / 'matrices_confusion_5_variables.csv',
    index=False
)


# ============================================================
# 16. VERIFICACION Y RESUMEN DE ARCHIVOS
# ============================================================

print("\nArchivos de test guardados:")
print("- resultados_test_5_variables.csv")
print("- matrices_confusion_5_variables.csv")

print("\nVerificacion de umbrales:")

print(
    "Gini > 0.60:",
    'CUMPLE' if metricas_test['Gini'] > 0.60
    else 'NO CUMPLE'
)

print(
    "KS > 40:",
    'CUMPLE' if metricas_test['KS'] > 40
    else 'NO CUMPLE'
)

print("\nArchivos de resultados guardados en:")
print("- comparacion_modelos_5_variables.csv")
print("- resultados_test_5_variables.csv")
print("- matrices_confusion_5_variables.csv")