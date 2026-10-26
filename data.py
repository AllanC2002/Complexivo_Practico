# Generación de datos sintéticos de riesgo crediticio (Ground Truth)
# 50.000 registros, relaciones lógicas, riesgo ponderado, default Bernoulli
# e inyección controlada de errores para probar un pipeline ETL.
# Este script SOLO genera y exporta el dataset; no entrena modelos ni limpia datos.

import numpy as np
import pandas as pd

np.random.seed(42)
n_registros = 50000

# 1. GENERACIÓN DE VARIABLES CON RELACIONES LÓGICAS

# Edad del titular
edad = np.random.randint(20, 71, n_registros)

# Ingreso mensual: distribución sesgada a la derecha
ingreso = np.random.lognormal(mean=6.5, sigma=0.6, size=n_registros) * 1.5

# Ratio deuda-ingreso (DTI)
dti = np.clip(
    np.random.normal(loc=0.35, scale=0.15, size=n_registros),
    0.01,
    0.99
)

# Tipo de vivienda
vivienda = np.random.choice(
    ['Propia', 'Alquilada', 'Familiar'],
    size=n_registros,
    p=[0.45, 0.40, 0.15]
)

# Monto solicitado relacionado con el ingreso
monto = np.clip(
    ingreso * np.random.uniform(2, 10, n_registros),
    500,
    50000
)

# Antigüedad laboral en meses: máximo según edad (se asume inicio laboral a los 18)
antiguedad_max_logica = (edad - 18) * 12
antiguedad = np.minimum(
    np.random.exponential(scale=36, size=n_registros),
    antiguedad_max_logica
).astype(int)
antiguedad = np.maximum(antiguedad, 0)

# Score de buró
score = np.clip(
    np.random.normal(loc=650, scale=80, size=n_registros),
    300,
    850
).astype(int)

# Mora histórica: menor score implica mayor probabilidad de registrar mora.
# La probabilidad se limita a un rango válido [0, 1].
prob_tener_mora = np.clip(1 - ((score - 300) / 550), 0, 1)
mora = np.where(
    np.random.rand(n_registros) < (prob_tener_mora * 0.6),
    np.random.randint(1, 13, n_registros),
    0
)

# Consultas recientes: se incrementa su media para scores bajos.
consultas_base = np.random.poisson(lam=1.0, size=n_registros)
consultas_extra = np.where(
    score < 600,
    np.random.poisson(lam=2.5, size=n_registros),
    0
)
consultas = consultas_base + consultas_extra

# 2. RIESGO LATENTE Y DEFAULT ESTOCÁSTICO (BERNOULLI)

# Normalización de variables para construir una ecuación de riesgo.
score_norm = (650 - score) / 100
mora_norm = mora / 3
dti_norm = (dti - 0.35) / 0.15
antiguedad_norm = (antiguedad - 36) / 36
ratio_monto_ingreso = monto / (ingreso + 1)
monto_norm = (ratio_monto_ingreso - 4) / 2
consultas_norm = consultas / 2
ingreso_norm = (ingreso - 1500) / 1000

viv_riesgo = np.where(
    vivienda == 'Alquilada',
    1.0,
    np.where(vivienda == 'Familiar', 0.5, 0.0)
)

# Ecuación de riesgo latente.
# La edad tiene peso directo cero. La antigüedad laboral sí puede estar
# relacionada indirectamente con la edad debido a la restricción lógica.
riesgo_oculto = (
    (1.6 * score_norm) +
    (1.4 * mora_norm) +
    (0.8 * dti_norm) -
    (0.8 * antiguedad_norm) +
    (0.7 * monto_norm) +
    (0.4 * consultas_norm) +
    (0.3 * viv_riesgo) -
    (0.2 * ingreso_norm) +
    (0.0 * ((edad - 45) / 15))
)

# Calibración mediante intercepto: se busca que la media de probabilidades
# individuales sea aproximadamente el 12 %, sin imponer una cuota fija
# de clientes morosos. Se conserva así la asignación estocástica Bernoulli.
def calibrar_intercepto(riesgo, tasa_objetivo=0.12, iteraciones=100):
    inferior, superior = -30.0, 30.0

    for _ in range(iteraciones):
        intercepto = (inferior + superior) / 2
        probabilidades = 1 / (1 + np.exp(-np.clip(riesgo + intercepto, -500, 500)))

        if probabilidades.mean() > tasa_objetivo:
            superior = intercepto
        else:
            inferior = intercepto

    return (inferior + superior) / 2

intercepto = calibrar_intercepto(riesgo_oculto, tasa_objetivo=0.12)
prob_calibrada = 1 / (
    1 + np.exp(-np.clip(riesgo_oculto + intercepto, -500, 500))
)

# El target se asigna de forma probabilística: no se usa un umbral fijo.
default = np.random.binomial(1, prob_calibrada)

# =====================================================================
# 3. ENSAMBLAJE DEL DATASET ORIGINAL
# =====================================================================

df = pd.DataFrame({
    'edad_titular': edad,
    'ingreso_mensual_usd': np.round(ingreso, 2),
    'ratio_deuda_ingreso': np.round(dti, 4),
    'antiguedad_laboral_meses': antiguedad,
    'tipo_vivienda': vivienda,
    'consultas_buro_ultimos_6m': consultas,
    'meses_maximo_mora_historica': mora,
    'score_buro_externo': score,
    'monto_credito_solicitado': np.round(monto, 2),
    'estado_default': default
})

# 4. INYECCIÓN CONTROLADA DE NULOS, OUTLIERS E INCONSISTENCIAS

# Valores nulos en variables predictoras
idx_score_nan = df.sample(frac=0.10, random_state=1).index
df.loc[idx_score_nan, 'score_buro_externo'] = np.nan

idx_ant_nan = df.sample(frac=0.05, random_state=2).index
df.loc[idx_ant_nan, 'antiguedad_laboral_meses'] = np.nan

idx_ing_nan = df.sample(frac=0.02, random_state=3).index
df.loc[idx_ing_nan, 'ingreso_mensual_usd'] = np.nan

# Proporción reducida de etiquetas desconocidas: 0,1 % (~50 de 50.000).
# Estas filas deberán tratarse por separado si luego se entrena un modelo.
idx_def_nan = df.sample(frac=0.001, random_state=4).index
df.loc[idx_def_nan, 'estado_default'] = np.nan

# Outliers en edad
outliers_edad = df.sample(15, random_state=5).index
df.loc[outliers_edad[:5], 'edad_titular'] = -1
df.loc[outliers_edad[5:10], 'edad_titular'] = 0
df.loc[outliers_edad[10:], 'edad_titular'] = 150

# Outliers en ingreso
outliers_ing = df.sample(5, random_state=6).index
df.loc[outliers_ing[:2], 'ingreso_mensual_usd'] = 9999999.0
df.loc[outliers_ing[2:], 'ingreso_mensual_usd'] = 1.0

# Montos de crédito negativos
outliers_monto = df.sample(5, random_state=7).index
df.loc[outliers_monto, 'monto_credito_solicitado'] = -5000.0

# Inconsistencias de formato y categoría en vivienda
idx_viv_propia = df[df['tipo_vivienda'] == 'Propia'].sample(
    frac=0.02, random_state=8
).index
df.loc[idx_viv_propia, 'tipo_vivienda'] = 'propia'

idx_viv_alq = df[df['tipo_vivienda'] == 'Alquilada'].sample(
    frac=0.02, random_state=9
).index
df.loc[idx_viv_alq, 'tipo_vivienda'] = ' ALQUILADA '

idx_viv_otro = df.sample(frac=0.01, random_state=10).index
df.loc[idx_viv_otro, 'tipo_vivienda'] = 'Otro'

# DTI negativo
idx_dti_neg = df.sample(10, random_state=11).index
df.loc[idx_dti_neg, 'ratio_deuda_ingreso'] = -0.15

# Inconsistencia lógica: antigüedad incompatible con la edad
idx_logic_age = df[
    (df['edad_titular'] >= 20) & (df['edad_titular'] <= 22)
].sample(10, random_state=12).index
df.loc[idx_logic_age, 'antiguedad_laboral_meses'] = 240

# 5. EXPORTACIÓN

ruta_salida = 'data/raw/clientes_solicitud_credito.csv'
df.to_csv(ruta_salida, index=False)

print(f'Dataset generado: {ruta_salida}')
print(f'Registros: {len(df):,}')
print(f'Tasa de default observada (excluyendo etiquetas nulas): '
      f'{df["estado_default"].mean():.2%}')
print(f'Etiquetas nulas: {df["estado_default"].isna().sum():,}')
print('Distribución de estado_default (incluye NaN):')
print(df['estado_default'].value_counts(dropna=False, normalize=True).sort_index())