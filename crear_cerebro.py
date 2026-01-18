import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

print("--- 1. CARGANDO DATOS ---")
# Cargamos tu Excel
try:
    df = pd.read_csv('Base_Datos_Cleaned.csv', sep=';')
except:
    print("ERROR: No encuentro el archivo 'Base_Datos_Cleaned.csv'.")
    exit()

# Limpieza rápida (La misma que diseñamos antes)
df['año_fabricacion'] = pd.to_numeric(df['año'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
df['antiguedad'] = 2026 - df['año_fabricacion']
df['ext_CV'] = pd.to_numeric(df['ext_CV'], errors='coerce')
df['kilometros'] = pd.to_numeric(df['kilometros'], errors='coerce')
df['precio_€'] = pd.to_numeric(df['precio_€'], errors='coerce')

# Filtros de calidad (Quitamos coches rotos y clásicos)
df_clean = df[
    (df['precio_€'] >= 1500) & (df['precio_€'] <= 140000) & 
    (df['kilometros'] > 100) & (df['kilometros'] < 400000) & 
    (df['ext_CV'] >= 50) & (df['ext_CV'] <= 600) &           
    (df['año_fabricacion'] >= 2000) &                        
    (df['antiguedad'].notna())
].copy()

# Función para limpiar nombres de modelos
def limpiar_modelo(row):
    marca = str(row['marca_busqueda']).lower().strip()
    m = str(row['modelo']).lower().strip()
    while m.startswith(marca): m = m[len(marca):].strip().lstrip("-:,.")
    words = m.split()
    if not words: return "Other"
    if words[0] in ['clase', 'serie', 'range'] and len(words)>1: return f"{words[0]} {words[1]}".title()
    return words[0].capitalize()

df_clean['modelo_agrupado'] = df_clean.apply(limpiar_modelo, axis=1)

# Nos quedamos con los 80 modelos más comunes
top_modelos = df_clean['modelo_agrupado'].value_counts().nlargest(80).index.tolist()
df_clean['modelo_agrupado'] = df_clean['modelo_agrupado'].apply(lambda x: x if x in top_modelos else 'Other')

# Preparamos X e y
X = df_clean[['kilometros', 'antiguedad', 'ext_CV', 'marca_busqueda', 'motor', 'modelo_agrupado']]
y = np.log1p(df_clean['precio_€'])

# Configuración de la IA (Pipeline)
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', RobustScaler())]), ['kilometros', 'antiguedad', 'ext_CV']),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='Unknown')), ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), ['marca_busqueda', 'motor', 'modelo_agrupado'])
    ])),
    ('regressor', SVR(kernel='rbf', C=10, epsilon=0.05)) # Configuración óptima
])

print("--- 2. ENTRENANDO IA (Espera unos segundos...) ---")
pipeline.fit(X, y)

# GUARDAMOS EL ARCHIVO .PKL
archivo_salida = 'modelo_svm_final_optimizado.pkl'
joblib.dump(pipeline, archivo_salida)

print(f"--- ¡LISTO! Se ha creado el archivo: {archivo_salida} ---")