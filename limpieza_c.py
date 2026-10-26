import pandas as pd
import numpy as np

# =====================================================================
# 1. CARGA Y CONFIGURACIÓN DE AUDITORÍA
# =====================================================================

df = pd.read_csv('clientes_solicitud_credito.csv')

auditoria = []

def registrar(variable, accion, cantidad, detalle=""):
    if cantidad > 0:
        auditoria.append({
            'Variable': variable,
            'Acción': accion,
            'Registros': int(cantidad),
            'Detalle': detalle
        })

columnas_requeridas = [
    'edad_titular',
    'ingreso_mensual_usd',
    'ratio_deuda_ingreso',
    'antiguedad_laboral_meses',
    'tipo_vivienda',
    'consultas_buro_ultimos_6m',
    'meses_maximo_mora_historica',
    'score_buro_externo',
    'monto_credito_solicitado',
    'estado_default'
]

faltantes = set(columnas_requeridas) - set(df.columns)

if faltantes:
    raise ValueError(f"Faltan columnas requeridas: {faltantes}")

print("Dimensiones iniciales:", df.shape)


# =====================================================================
# 2. ELIMINACIÓN DE REGISTROS SIN TARGET
# =====================================================================

nulos_target = int(df['estado_default'].isnull().sum())

df = df.dropna(subset=['estado_default']).copy()

registrar(
    'estado_default',
    'Registros eliminados por target nulo',
    nulos_target
)

if not df['estado_default'].isin([0, 1]).all():
    raise ValueError("El target contiene valores distintos de 0 y 1.")


# =====================================================================
# 3. ESTANDARIZACIÓN DE TIPO DE VIVIENDA
# =====================================================================

vivienda_original = df['tipo_vivienda'].astype('string')

vivienda_normalizada = (
    vivienda_original.str.strip().str.lower()
)

cambios_formato = int((
    vivienda_original.fillna('<NULO>')
    != vivienda_normalizada.fillna('<NULO>')
).sum())

registrar(
    'tipo_vivienda',
    'Formato estandarizado',
    cambios_formato
)

mapeo_vivienda = {
    'propia': 'propia',
    'alquilada': 'alquilada',
    'familiar': 'familiar'
}

desconocidos = int((
    vivienda_normalizada.notna()
    & ~vivienda_normalizada.isin(mapeo_vivienda.keys())
).sum())

nulos_vivienda = int(vivienda_normalizada.isnull().sum())

df['tipo_vivienda'] = (
    vivienda_normalizada
    .map(mapeo_vivienda)
    .fillna('desconocido')
)

registrar(
    'tipo_vivienda',
    'Categorías no reconocidas asignadas a desconocido',
    desconocidos
)

registrar(
    'tipo_vivienda',
    'Nulos asignados a desconocido',
    nulos_vivienda
)


# =====================================================================
# 4. CORRECCIÓN DE INCONSISTENCIAS Y OUTLIERS
# =====================================================================

# 4.1 Edad

edades_validas = df.loc[
    df['edad_titular'].between(18, 100),
    'edad_titular'
]

if edades_validas.empty:
    raise ValueError("No existen edades válidas para calcular la mediana.")

mediana_edad = edades_validas.median()

edad_invalida = (
    df['edad_titular'].notna()
    & ~df['edad_titular'].between(18, 100)
)

nulos_edad = int(df['edad_titular'].isnull().sum())
edades_corregidas = int(edad_invalida.sum())

df.loc[
    ~df['edad_titular'].between(18, 100),
    'edad_titular'
] = mediana_edad

registrar(
    'edad_titular',
    'Valores fuera de rango corregidos',
    edades_corregidas,
    f'Mediana = {mediana_edad:.2f}'
)

registrar(
    'edad_titular',
    'Nulos imputados',
    nulos_edad,
    f'Mediana = {mediana_edad:.2f}'
)


# 4.2 Ingresos

ingresos_bajos = int((
    df['ingreso_mensual_usd'].notna()
    & (df['ingreso_mensual_usd'] < 100)
).sum())

df.loc[
    df['ingreso_mensual_usd'] < 100,
    'ingreso_mensual_usd'
] = np.nan

registrar(
    'ingreso_mensual_usd',
    'Ingresos inferiores a $100 convertidos en nulos',
    ingresos_bajos
)

ingresos_validos = df['ingreso_mensual_usd'].dropna()

if ingresos_validos.empty:
    raise ValueError("No existen ingresos válidos.")

limite_ingreso = ingresos_validos.quantile(0.99)
mediana_ingreso = ingresos_validos.median()

ingresos_extremos = int((
    df['ingreso_mensual_usd'] > limite_ingreso
).sum())

df.loc[
    df['ingreso_mensual_usd'] > limite_ingreso,
    'ingreso_mensual_usd'
] = limite_ingreso

registrar(
    'ingreso_mensual_usd',
    'Valores limitados al percentil 99',
    ingresos_extremos,
    f'Límite = ${limite_ingreso:.2f}'
)


# 4.3 Monto solicitado

montos_negativos = int((
    df['monto_credito_solicitado'].notna()
    & (df['monto_credito_solicitado'] < 0)
).sum())

df.loc[
    df['monto_credito_solicitado'] < 0,
    'monto_credito_solicitado'
] = np.nan

registrar(
    'monto_credito_solicitado',
    'Montos negativos convertidos en nulos',
    montos_negativos
)


# 4.4 Ratio deuda-ingreso

dti_negativos = int((
    df['ratio_deuda_ingreso'].notna()
    & (df['ratio_deuda_ingreso'] < 0)
).sum())

df.loc[
    df['ratio_deuda_ingreso'] < 0,
    'ratio_deuda_ingreso'
] = np.nan

registrar(
    'ratio_deuda_ingreso',
    'Valores negativos convertidos en nulos',
    dti_negativos
)


# 4.5 Variables de conteo

columnas_conteo = [
    'antiguedad_laboral_meses',
    'consultas_buro_ultimos_6m',
    'meses_maximo_mora_historica'
]

for columna in columnas_conteo:

    negativos = int((
        df[columna].notna() & (df[columna] < 0)
    ).sum())

    df.loc[df[columna] < 0, columna] = np.nan

    registrar(
        columna,
        'Valores negativos convertidos en nulos',
        negativos
    )


# 4.6 Score de buró

scores_invalidos = int((
    df['score_buro_externo'].notna()
    & ~df['score_buro_externo'].between(300, 850)
).sum())

df.loc[
    ~df['score_buro_externo'].between(300, 850),
    'score_buro_externo'
] = np.nan

registrar(
    'score_buro_externo',
    'Scores fuera del rango 300-850 convertidos en nulos',
    scores_invalidos
)


# 4.7 Coherencia entre edad y antigüedad laboral

max_meses_posibles = (df['edad_titular'] - 18) * 12

mas_antiguedad = (
    df['antiguedad_laboral_meses'].notna()
    & (
        df['antiguedad_laboral_meses']
        > max_meses_posibles
    )
)

antiguedades_corregidas = int(mas_antiguedad.sum())

df.loc[
    mas_antiguedad,
    'antiguedad_laboral_meses'
] = max_meses_posibles[mas_antiguedad]

registrar(
    'antiguedad_laboral_meses',
    'Antigüedades incompatibles con la edad corregidas',
    antiguedades_corregidas
)


# =====================================================================
# 5. IMPUTACIÓN DE VALORES NULOS
# =====================================================================

def imputar_mediana(columna, mediana=None, respaldo=None):

    if mediana is None:
        mediana = df[columna].median()

    if pd.isna(mediana):
        if respaldo is not None:
            mediana = respaldo
        else:
            raise ValueError(
                f"No existen valores válidos para {columna}."
            )

    nulos = int(df[columna].isnull().sum())

    df[columna] = df[columna].fillna(mediana)

    registrar(
        columna,
        'Nulos imputados con la mediana',
        nulos,
        f'Valor utilizado = {mediana:.2f}'
    )


imputar_mediana('ingreso_mensual_usd', mediana_ingreso)
imputar_mediana('antiguedad_laboral_meses', respaldo=0)
imputar_mediana('score_buro_externo')
imputar_mediana('ratio_deuda_ingreso')
imputar_mediana('monto_credito_solicitado')
imputar_mediana('consultas_buro_ultimos_6m', respaldo=0)
imputar_mediana('meses_maximo_mora_historica', respaldo=0)


# =====================================================================
# 6. VALIDACIÓN FINAL Y AJUSTES
# =====================================================================

# Comprobar nuevamente la antigüedad laboral.

max_meses_posibles = (df['edad_titular'] - 18) * 12

ajustes_finales = int((
    df['antiguedad_laboral_meses'] > max_meses_posibles
).sum())

df['antiguedad_laboral_meses'] = np.minimum(
    df['antiguedad_laboral_meses'],
    max_meses_posibles
)

registrar(
    'antiguedad_laboral_meses',
    'Ajustes finales de consistencia',
    ajustes_finales
)


# Redondear variables enteras.

columnas_enteras = [
    'edad_titular',
    'antiguedad_laboral_meses',
    'consultas_buro_ultimos_6m',
    'meses_maximo_mora_historica',
    'score_buro_externo'
]

for columna in columnas_enteras:

    valores_no_enteros = int((
        ~np.isclose(df[columna], df[columna].round())
    ).sum())

    df[columna] = df[columna].round().astype(int)

    registrar(
        columna,
        'Valores redondeados a enteros',
        valores_no_enteros
    )


# =====================================================================
# 7. REPORTE DE AUDITORÍA Y EXPORTACIÓN
# =====================================================================

print("\n" + "=" * 75)
print("AUDITORÍA DE LIMPIEZA")
print("=" * 75)

if auditoria:
    reporte = pd.DataFrame(auditoria)
    print(reporte.to_string(index=False))
else:
    print("No se registraron cambios.")

print("\n" + "-" * 75)
print("VALIDACIÓN FINAL")
print("-" * 75)

print("Dimensiones finales:", df.shape)
print("Valores nulos restantes:", int(df.isnull().sum().sum()))
print("Target: etiquetas inválidas:",
      int((~df['estado_default'].isin([0, 1])).sum()))
print("Edades fuera de 18-100:",
      int((~df['edad_titular'].between(18, 100)).sum()))
print("Montos negativos:",
      int((df['monto_credito_solicitado'] < 0).sum()))
print("DTI negativos:",
      int((df['ratio_deuda_ingreso'] < 0).sum()))
print("Scores fuera de 300-850:",
      int((~df['score_buro_externo'].between(300, 850)).sum()))

print("Tasa de default:",
      f"{df['estado_default'].mean():.2%}")

df.to_csv('clientes_solicitud_credito_clean.csv', index=False)

print("\nDataset exportado: clientes_solicitud_credito_clean.csv")
print("Columnas conservadas:", len(df.columns))