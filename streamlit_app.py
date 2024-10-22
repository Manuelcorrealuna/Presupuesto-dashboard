import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
import textwrap
import openai
import pydot
from networkx.drawing.nx_pydot import graphviz_layout  # Usamos pydot en lugar de pygraphviz

# Cargar los datos
gastos = pd.read_csv('data/Gastos.csv', encoding='latin1', delimiter=';')
recursos_humanos = pd.read_csv('data/RecursosHumanos.csv', encoding='latin1', delimiter=';')
organigrama_df = pd.read_csv('data/Estructura.csv', encoding='latin1', delimiter=';')

# Filtrar las columnas necesarias para el organigrama
org_data = organigrama_df[['Unidad', 'Reporta_a', 'Jurisdiccion']].copy()

# Función para filtrar los datos
def filtrar_datos(df, columna, valor):
    return df if valor == 'Todas' else df[df[columna] == valor]

# Filtro único de jurisdicción
jurisdiccion = st.sidebar.selectbox('Seleccionar Jurisdicción', ['Todas'] + list(gastos['JURISDICCION'].dropna().unique()))

# Aplicar filtro a todos los DataFrames
gastos_filtrados = filtrar_datos(gastos, 'JURISDICCION', jurisdiccion)
recursos_humanos_filtrados = filtrar_datos(recursos_humanos, 'JURISDICCION', jurisdiccion)
organigrama_filtrado = filtrar_datos(org_data, 'Jurisdiccion', jurisdiccion)


# Función para ajustar el texto en varias líneas si es muy largo
def wrap_text(text, width):
    return "\n".join(textwrap.wrap(text, width))

# Función para crear y mostrar el organigrama con ajuste de leyendas largas
def plot_hierarchical_org_chart(jurisdiction_data):
    # Crear un grafo dirigido
    G = nx.DiGraph()
    
    # Añadir nodos y aristas basados en el filtro
    for _, row in jurisdiction_data.iterrows():
        if pd.notna(row['Reporta_a']):
            G.add_edge(row['Reporta_a'], row['Unidad'])
        else:
            G.add_node(row['Unidad'])
    
    # Usar layout jerárquico con pydot
    pos = graphviz_layout(G, prog='dot')  # Usamos pydot para el layout
    
    # Ajustar tamaño de la figura
    plt.figure(figsize=(16, 10))
    
    # Dibujar nodos, aristas y etiquetas
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='lightblue', alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='gray')
    
    # Ajustar etiquetas con texto dividido en varias líneas
    labels = {node: wrap_text(node, 20) for node in G.nodes()}  # Aquí ajustas la longitud máxima de cada línea (20)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold', 
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Añadir título y desactivar ejes
    plt.title(f"Organigrama - Jurisdicción: {jurisdiccion if jurisdiccion != 'Todas' else 'Todos'}", fontsize=14)
    plt.axis('off')
    
    # Mostrar en Streamlit
    st.pyplot(plt)
    plt.clf()

# Mostrar el organigrama filtrado
st.title("Organigrama de la Estructura Organizacional")
plot_hierarchical_org_chart(organigrama_filtrado)

# Aquí sigue el código para los gráficos de Gastos y Recursos Humanos sin modificaciones
# ...

# Función para analizar los datasets filtrados usando la API de chat
def obtener_analisis(datos_gastos, datos_rrhh, datos_organigrama):
    # Limitar el tamaño de los datasets
    datos_gastos = limitar_dataset(datos_gastos)
    datos_rrhh = limitar_dataset(datos_rrhh)
    datos_organigrama = limitar_dataset(datos_organigrama)
    
    # Preparar los datos para enviar como prompt
    prompt = f"""
    Soy funcionario en el estado nacional de argentina, y quiero que seas mi analista de negocio para que puedas hacer consultoría estratégica.
    Proporciona un análisis detallado de los gastos, recursos humanos y estructura organizacional, considerando las diferencias entre los distintos años.
    
    Datos de Gastos:
    {datos_gastos.to_string()}
    
    Datos de Recursos Humanos:
    {datos_rrhh.to_string()}
    
    Datos de Estructura Organizacional:
    {datos_organigrama.to_string()}
    
    ¿Cuáles son las observaciones más importantes? ¿Qué tendencias o recomendaciones se podrían extraer de estos datos?
    """
    
    # Llamada a la nueva API de OpenAI utilizando el modelo de chat 'gpt-3.5-turbo' o 'gpt-4'
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # O usa 'gpt-4' si tienes acceso
        messages=[
            {"role": "system", "content": "Actúa como un experto Business Analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,  # Ajusta el número de tokens según la longitud de la respuesta
        temperature=0.7,
    )
    
    # Acceder al contenido del mensaje desde el objeto de respuesta
    return response.choices[0].message['content']

# Crear un botón en la interfaz de Streamlit
if st.button("Analizar datos con OpenAI"):
    # Ejecutar la función que analiza los datos
    analisis = obtener_analisis(gastos_filtrados, recursos_humanos_filtrados, organigrama_filtrado)
    
    # Mostrar el análisis en la interfaz
    st.subheader("Análisis realizado por OpenAI")
    st.write(analisis)
