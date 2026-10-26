import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar la data limpia (la capa "clean" que guardamos)
df = pd.read_csv('clientes_solicitud_credito_clean.csv')

# =====================================================================
# A. ANÁLISIS DE CORRELACIÓN LINEAL (Pearson)
# =====================================================================
print("--- Generando Matriz de Correlación ---")
# Seleccionamos solo las variables numéricas
num_cols = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(10, 8))
matriz_corr = df[num_cols].corr()

# Generar un mapa de calor (Heatmap) para visualizarlo fácilmente
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0)
plt.title("Matriz de Correlación - Variables Numéricas")
plt.tight_layout()
# plt.show() # Descomenta para ver el gráfico en tu entorno
plt.savefig("matriz_correlacion.png") 

# =====================================================================
# B. PRUEBAS DE PODER PREDICTIVO (Information Value - IV)
# =====================================================================
print("\n--- Calculando Information Value (IV) por Variable ---")

def calcular_iv(df, feature, target):
    """
    Calcula el Information Value (IV) dividiendo las variables continuas en deciles.
    IV = Suma de ( % Buenos - % Malos ) * WoE
    """
    # Si la variable es numérica y tiene muchos valores únicos, la dividimos en 10 rangos (deciles)
    if pd.api.types.is_numeric_dtype(df[feature]) and df[feature].nunique() > 10:
        # Se usa qcut para crear bins con la misma cantidad de observaciones
        binned = pd.qcut(df[feature], q=10, duplicates='drop')
    else:
        binned = df[feature] # Para categóricas o discretas con pocos valores

    # Agrupar para contar totales y sumar los defaults (malos)
    grouped = df.groupby(binned, observed=False)[target].agg(['count', 'sum'])
    grouped = grouped.rename(columns={'count': 'total', 'sum': 'malos'})
    grouped['buenos'] = grouped['total'] - grouped['malos']
    
    # Manejar posibles divisiones por cero agregando una constante minúscula
    epsilon = 1e-6
    grouped['malos'] = grouped['malos'].replace(0, epsilon)
    grouped['buenos'] = grouped['buenos'].replace(0, epsilon)
    
    # Calcular distribución porcentual
    dist_malos = grouped['malos'] / grouped['malos'].sum()
    dist_buenos = grouped['buenos'] / grouped['buenos'].sum()
    
    # Calcular WoE y luego el IV del segmento
    woe = np.log(dist_buenos / dist_malos)
    iv_segmento = (dist_buenos - dist_malos) * woe
    
    # El IV de la variable es la suma de los IV de sus segmentos
    return iv_segmento.sum()

# Iterar sobre todas las variables predictoras (excluyendo el target)
variables_predictoras = [col for col in df.columns if col != 'estado_default']
resultados_iv = []

for var in variables_predictoras:
    iv_val = calcular_iv(df, var, 'estado_default')
    resultados_iv.append({'Variable': var, 'Information Value (IV)': iv_val})

# Mostrar resultados ordenados de mayor a menor poder predictivo
df_iv = pd.DataFrame(resultados_iv).sort_values(by='Information Value (IV)', ascending=False)
df_iv['Poder Predictivo'] = pd.cut(df_iv['Information Value (IV)'], 
                                   bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf], 
                                   labels=['Inútil', 'Débil', 'Medio', 'Fuerte', 'Alto'])

print(df_iv.to_string(index=False))