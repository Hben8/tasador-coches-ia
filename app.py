import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Tasador IA de Coches",
    page_icon="🚗",
    layout="centered"
)

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# 2. CARGAR RECURSOS (MODELO Y DATOS)
# ============================================
@st.cache_resource
def cargar_recursos():
    # A) Cargar Modelo
    modelo_ia = None
    try:
        # Asegúrate de que este es el archivo generado por el script nuevo
        modelo_ia = joblib.load('modelo_svm_final_optimizado.pkl')
    except:
        st.error("Error crítico: No se encuentra el archivo .pkl del modelo.")

    # B) Cargar Datos para los Menús
    dict_modelos = {}
    try:
        df = pd.read_csv('19_01_2026.csv', sep=';')
        
        # Lógica de limpieza para obtener modelos limpios (Igual que en el entrenamiento)
        def limpiar_modelo_menu(row):
            marca = str(row['marca_busqueda']).lower().strip()
            m = str(row['modelo']).lower().strip()
            if m.startswith(marca): m = m[len(marca):].strip().lstrip("-:,.")
            words = m.split()
            if not words: return "Other"
            # Compuestos
            if len(words) > 1 and words[0] in ['clase', 'serie', 'range', 'grand']:
                return f"{words[0]} {words[1]}".title()
            if len(words) > 1 and words[0] == 'rav' and words[1] == '4':
                return "Rav4"
            return words[0].capitalize()

        df['modelo_menu'] = df.apply(limpiar_modelo_menu, axis=1)
        
        # Crear diccionario {Marca: [Lista de Modelos]}
        marcas_disponibles = df['marca_busqueda'].unique()
        
        for marca in marcas_disponibles:
            # Filtramos modelos de esa marca y los ordenamos alfabéticamente
            modelos = sorted(df[df['marca_busqueda'] == marca]['modelo_menu'].unique().tolist())
            dict_modelos[marca] = modelos
            
    except Exception as e:
        st.error(f"No se pudo cargar la base de datos para los menús: {e}")
    
    return modelo_ia, dict_modelos

pipeline, modelos_por_marca = cargar_recursos()

# ============================================
# 3. LÓGICA DE PREDICCIÓN
# ============================================
def procesar_datos(marca, modelo, motor, km, cv, anio):
    # La limpieza de texto ya viene hecha del menú, pero mantenemos la estructura
    anio_actual = 2026
    antiguedad = anio_actual - anio
    
    # Preparamos el DataFrame para la IA
    datos = pd.DataFrame({
        'kilometros': [km],
        'antiguedad': [antiguedad],
        'ext_CV': [cv],
        'marca_busqueda': [marca], 
        'motor': [motor],
        # CORRECCIÓN AQUÍ: Antes decía 'modelo_agrupado', ahora debe ser 'modelo_final'
        'modelo_final': [modelo] 
    })
    return datos

# ============================================
# 4. INTERFAZ DE USUARIO
# ============================================
st.title("🚗 Tasador Inteligente de Vehículos")
st.markdown("### Descubre el valor de mercado de tu coche usando IA")
st.write("Selecciona las características exactas del vehículo.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    # 1. SELECCIÓN DE MARCA
    # Obtenemos la lista de marcas del diccionario que creamos al inicio
    lista_marcas = sorted(list(modelos_por_marca.keys())) if modelos_por_marca else ["Error al cargar"]
    marca = st.selectbox("Marca", lista_marcas)
    
    # 2. SELECCIÓN DE MODELO (DINÁMICO)
    # Si hay marcas cargadas, buscamos los modelos de la marca seleccionada
    opciones_modelo = modelos_por_marca.get(marca, ["Otros"]) if modelos_por_marca else []
    modelo = st.selectbox("Modelo", opciones_modelo)
    
    motor = st.selectbox("Tipo de Motor", ["Diésel", "Gasolina", "Híbrido (HEV/MHEV)", "Eléctrico (BEV)", "Híbrido Enchufable (PHEV)", "GLP", "GNC"])

with col2:
    anio = st.number_input("Año de Matriculación", min_value=1990, max_value=2026, value=2019, step=1)
    km = st.number_input("Kilometraje", min_value=0, max_value=500000, value=80000, step=1000)
    cv = st.number_input("Potencia (CV)", min_value=50, max_value=800, value=150, step=10)

st.markdown("---")

if st.button("CALCULAR PRECIO"):
    if not pipeline:
        st.error("Error: No se ha cargado el modelo IA.")
    else:
        # Procesar
        df_input = procesar_datos(marca, modelo, motor, km, cv, anio)
        
        try:
            with st.spinner('Consultando a la IA...'):
                pred_log = pipeline.predict(df_input)
                precio = np.expm1(pred_log)[0]
            
            st.success("¡Tasación Completada!")
            st.metric(label="Valor Estimado de Mercado", value=f"{precio:,.2f} €")
            
            error_margen = 1270 
            st.info(f"Rango estimado: **{precio - error_margen:,.0f} €** - **{precio + error_margen:,.0f} €**")
            
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")