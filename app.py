import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ============================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Tasador IA de Coches",
    page_icon="🚗",
    layout="centered"
)

# Estilo CSS para que se vea profesional
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# 2. CARGAR EL CEREBRO (MODELO)
# ============================================
@st.cache_resource
def cargar_modelo():
    try:
        # Asegúrate de que el nombre coincida con tu archivo
        return joblib.load('modelo_svm_final_optimizado.pkl')
    except:
        return None

pipeline = cargar_modelo()

# ============================================
# 3. LÓGICA DE LIMPIEZA (LA MISMA DEL ENTRENAMIENTO)
# ============================================
def procesar_datos(marca, modelo, motor, km, cv, anio):
    # 1. Calcular Antigüedad
    anio_actual = 2026 # O datetime.now().year
    antiguedad = anio_actual - anio
    
    # 2. Limpieza de Texto del Modelo
    marca = str(marca).lower().strip()
    modelo_str = str(modelo).lower().strip()
    
    # Quitamos la marca del modelo si se repite
    while modelo_str.startswith(marca):
        modelo_str = modelo_str[len(marca):].strip().lstrip("-:,.")
        
    words = modelo_str.split()
    modelo_clean = words[0].capitalize() if words else "Other"
    
    # Manejo de compuestos (Clase A, Serie 3)
    if words and words[0] in ['clase', 'serie', 'class', 'range']:
        if len(words) > 1:
            modelo_clean = f"{words[0]} {words[1]}".title()
    
    # NOTA: En una app real, aquí deberíamos verificar si 'modelo_clean'
    # está en la lista de los top 80 que usó el modelo. 
    # Por simplicidad, asumimos que la IA manejará el 'Other' internamente 
    # si el OneHotEncoder estaba configurado con handle_unknown='ignore'.
    
    # 3. Crear DataFrame
    datos = pd.DataFrame({
        'kilometros': [km],
        'antiguedad': [antiguedad],
        'ext_CV': [cv],
        'marca_busqueda': [marca], # Ojo: La IA espera la marca tal cual se entrenó
        'motor': [motor],
        'modelo_agrupado': [modelo_clean]
    })
    
    return datos

# ============================================
# 4. INTERFAZ DE USUARIO (FRONTEND)
# ============================================
st.title("🚗 Tasador Inteligente de Vehículos")
st.markdown("### Descubre el valor de mercado de tu coche usando IA")
st.write("Esta aplicación utiliza un modelo **Support Vector Machine (SVM)** optimizado para estimar el precio real.")

st.markdown("---")

# Columnas para organizar inputs
col1, col2 = st.columns(2)

with col1:
    marca = st.selectbox("Marca", ["Audi", "BMW", "Mercedes-Benz", "Volkswagen", "Seat", "Renault", "Peugeot", "Ford", "Toyota", "Volvo", "Citroen", "Porsche", "Hyundai", "Kia", "Fiat", "Nissan", "Mini", "Land", "Jaguar", "Mazda", "Opel"])
    modelo = st.text_input("Modelo (Ej: A3, Golf, Serie 3)", placeholder="Escribe el modelo...")
    motor = st.selectbox("Tipo de Motor", ["Diésel", "Gasolina", "Híbrido (HEV/MHEV)", "Eléctrico (BEV)", "Híbrido Enchufable (PHEV)", "GLP", "GNC"])

with col2:
    anio = st.number_input("Año de Matriculación", min_value=1990, max_value=2026, value=2019, step=1)
    km = st.number_input("Kilometraje", min_value=0, max_value=500000, value=80000, step=1000)
    cv = st.number_input("Potencia (CV)", min_value=50, max_value=800, value=150, step=10)

st.markdown("---")

# Botón de Predicción
if st.button("CALCULAR PRECIO"):
    if not pipeline:
        st.error("Error: No se encuentra el archivo del modelo .pkl")
    elif not modelo:
        st.warning("Por favor, introduce un modelo.")
    else:
        # Procesar
        df_input = procesar_datos(marca, modelo, motor, km, cv, anio)
        
        # Predecir
        try:
            with st.spinner('Consultando a la IA...'):
                pred_log = pipeline.predict(df_input)
                precio = np.expm1(pred_log)[0]
            
            st.success("¡Tasación Completada!")
            
            # Mostrar resultado en grande
            st.metric(label="Valor Estimado de Mercado", value=f"{precio:,.2f} €")
            
            # Mostrar rango de error
            error_margen = 1270 # Tu mediana de error
            st.info(f"Rango de precio sugerido: **{precio - error_margen:,.0f} €** - **{precio + error_margen:,.0f} €**")
            
        except Exception as e:
            st.error(f"Ocurrió un error en la predicción: {e}")
            st.write("Detalles técnicos:", df_input)

# Footer
st.markdown("---")
st.caption("Desarrollado para Proyecto Universitario | Modelo SVM v1.0")