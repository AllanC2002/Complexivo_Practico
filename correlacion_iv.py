import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar los datos limpios
df = pd.read_csv('data/processed/clientes_solicitud_credito_clean.csv')

# A. ANALISIS DE CORRELACION LINEAL (PEARSON)
print("--- Generando Matriz de Correlacion ---")

num_cols = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(10, 8))
matriz_corr = df[num_cols].corr()

sns.heatmap(
    matriz_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0
)

plt.title("Matriz de Correlacion - Variables Numericas")
plt.tight_layout()
plt.savefig("reports/figures/matriz_correlacion.png", dpi=300, bbox_inches="tight")
plt.close()

print("Matriz de correlacion exportada: matriz_correlacion.png")


print("\n--- Calculando Information Value por Variable ---")


def calcular_iv(df, feature, target):
    """
    Calcula el Information Value de una variable.
    Las variables numericas con mas de 10 valores unicos
    se dividen en hasta 10 intervalos.
    """

    # Dividir variables numericas continuas en intervalos
    if (
        pd.api.types.is_numeric_dtype(df[feature])
        and df[feature].nunique() > 10
    ):
        binned = pd.qcut(
            df[feature],
            q=10,
            duplicates='drop'
        )
    else:
        binned = df[feature]

    # Contar total de registros y casos con incumplimiento
    grouped = df.groupby(
        binned,
        observed=False
    )[target].agg(['count', 'sum'])

    grouped = grouped.rename(
        columns={
            'count': 'total',
            'sum': 'malos'
        }
    )

    grouped['buenos'] = grouped['total'] - grouped['malos']

    # Evitar divisiones por cero
    epsilon = 1e-6

    grouped['malos'] = grouped['malos'].replace(0, epsilon)
    grouped['buenos'] = grouped['buenos'].replace(0, epsilon)

    # Distribucion de buenos y malos
    dist_malos = grouped['malos'] / grouped['malos'].sum()
    dist_buenos = grouped['buenos'] / grouped['buenos'].sum()

    # Calcular WoE e IV
    woe = np.log(dist_buenos / dist_malos)
    iv_segmento = (dist_buenos - dist_malos) * woe

    return iv_segmento.sum()


# C. CALCULAR IV PARA CADA VARIABLE
variables_predictoras = [
    col for col in df.columns
    if col != 'estado_default'
]

resultados_iv = []

for var in variables_predictoras:
    iv_val = calcular_iv(df, var, 'estado_default')

    resultados_iv.append({
        'Variable': var,
        'Information Value (IV)': iv_val
    })

# Ordenar de mayor a menor IV
df_iv = pd.DataFrame(resultados_iv)

df_iv = df_iv.sort_values(
    by='Information Value (IV)',
    ascending=False
).reset_index(drop=True)

# Clasificar el poder predictivo
df_iv['Poder Predictivo'] = pd.cut(
    df_iv['Information Value (IV)'],
    bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
    labels=['Muy Débil', 'Débil', 'Medio', 'Fuerte', 'Alto']
)


# D. MOSTRAR RESULTADOS POR CONSOLA
print("\n--- RESULTADOS DE INFORMATION VALUE (IV) ---\n")

print(
    df_iv.to_string(
        index=False,
        formatters={
            'Information Value (IV)': lambda x: f'{x:.4f}'
        }
    )
)


# E. EXPORTAR LA TABLA IV COMO IMAGEN PNG

# Preparar los datos que apareceran en la imagen
df_imagen = df_iv.copy()

df_imagen['Information Value (IV)'] = (
    df_imagen['Information Value (IV)'].map(
        lambda x: f'{x:.4f}'
    )
)

# Ajustar el tamaño segun la cantidad de variables
alto = max(3.5, len(df_imagen) * 0.55 + 1.5)

fig, ax = plt.subplots(figsize=(13, alto))

# Ocultar los ejes
ax.axis('off')

# Crear titulo
ax.set_title(
    'Valores de Information Value (IV) por Variable',
    fontsize=15,
    fontweight='bold',
    pad=20
)

# Crear tabla
tabla = ax.table(
    cellText=df_imagen.values,
    colLabels=df_imagen.columns,
    cellLoc='center',
    colLoc='center',
    colWidths=[0.50, 0.25, 0.25],
    loc='center'
)

# Configurar fuente y espaciado
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 1.6)

# Dar formato a las celdas
for (fila, columna), celda in tabla.get_celld().items():

    celda.set_edgecolor('#C8D0D9')

    if fila == 0:
        # Encabezado
        celda.set_facecolor('#243B53')
        celda.set_text_props(
            color='white',
            weight='bold'
        )
    else:
        # Filas alternadas
        if fila % 2 == 0:
            celda.set_facecolor('#F0F4F8')
        else:
            celda.set_facecolor('white')

        # Alinear los nombres de las variables a la izquierda
        if columna == 0:
            celda.set_text_props(ha='left')

# Guardar la imagen con buena resolucion
plt.savefig(
    'reports/figures/tabla_valores_iv.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.close(fig)

print("\nTabla IV exportada correctamente: tabla_valores_iv.png")
print("Total de variables analizadas:", len(df_iv))