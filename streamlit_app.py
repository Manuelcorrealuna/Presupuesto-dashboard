import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
import textwrap
import openai

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
    
    # Usar un layout nativo de networkx (spring_layout en este caso)
    pos = nx.spring_layout(G, k=0.5, iterations=50)  # Ajusta 'k' para controlar la separación de los nodos
    
    # Ajustar tamaño de la figura
    plt.figure(figsize=(16, 10))
    
    # Dibujar nodos, aristas y etiquetas
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='lightblue', alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='gray')
    
    # Ajustar etiquetas con texto dividido en varias líneas
    labels = {node: wrap_text(node, 20) for node in G.nodes()}
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

# Crear el gráfico de Gastos
gastos_agrupados = gastos_filtrados.groupby(['Año', 'Objeto Detalle'])['Monto'].sum().reset_index()
gastos_total_por_anio = gastos_agrupados.groupby('Año')['Monto'].sum().reset_index().rename(columns={'Monto': 'Total'})
gastos_agrupados = gastos_agrupados.merge(gastos_total_por_anio, on='Año')
gastos_agrupados['Porcentaje'] = gastos_agrupados['Monto'] / gastos_agrupados['Total'] * 100
gastos_agrupados['Monto'] = gastos_agrupados['Monto'] / 1_000_000_000  # Conversión a miles de millones

# Ordenar de menor a mayor y obtener el orden de leyendas para Gastos, invertido
gastos_totales = gastos_agrupados.groupby('Objeto Detalle')['Monto'].sum().sort_values(ascending=True)
leyendas_gastos = gastos_totales.index[::-1]  # Ordenado inverso

fig_gastos = make_subplots(rows=1, cols=2, subplot_titles=("Gastos Totales", "Gastos Porcentuales"))

# Asignar colores a cada Objeto Detalle para evitar tonos oscuros
color_map = {}
for i, objeto in enumerate(leyendas_gastos):
    color = f'rgba({(100 + i*25) % 255}, {(150 + i*35) % 255}, {(200 + i*45) % 255}, 0.8)'
    color_map[objeto] = color

for objeto in leyendas_gastos:
    objeto_data = gastos_agrupados[gastos_agrupados['Objeto Detalle'] == objeto]
    fig_gastos.add_trace(
        go.Bar(x=objeto_data['Año'], y=objeto_data['Monto'], name=objeto, legendgroup=objeto, 
               showlegend=True, text=objeto_data['Monto'].apply(lambda x: f'{x:,.2f} MM'), 
               hovertext=objeto_data['Objeto Detalle'], marker_color=color_map[objeto]),
        row=1, col=1
    )
    fig_gastos.add_trace(
        go.Bar(x=objeto_data['Año'], y=objeto_data['Porcentaje'], name=objeto, legendgroup=objeto, 
               showlegend=False, text=objeto_data['Porcentaje'].apply(lambda x: f'{x:.0f}%'), 
               hovertext=objeto_data['Objeto Detalle'], marker_color=color_map[objeto]),
        row=1, col=2
    )

fig_gastos.update_layout(barmode='stack', title="Gastos por Objeto Detalle", height=600)
st.plotly_chart(fig_gastos)

# Crear el gráfico de Recursos Humanos
rrhh_agrupados = recursos_humanos_filtrados.groupby(['Año', 'JURISDICCION'])['Cantidad Cargos'].sum().reset_index()
rrhh_total_por_anio = rrhh_agrupados.groupby('Año')['Cantidad Cargos'].sum().reset_index().rename(columns={'Cantidad Cargos': 'Total'})
rrhh_agrupados = rrhh_agrupados.merge(rrhh_total_por_anio, on='Año')
rrhh_agrupados['Porcentaje'] = rrhh_agrupados['Cantidad Cargos'] / rrhh_agrupados['Total'] * 100

# Ordenar de menor a mayor y obtener el orden de leyendas para Recursos Humanos, invertido
rrhh_totales = rrhh_agrupados.groupby('JURISDICCION')['Cantidad Cargos'].sum().sort_values(ascending=True)
leyendas_rrhh = rrhh_totales.index[::-1]  # Ordenado inverso

fig_rrhh = make_subplots(rows=1, cols=2, subplot_titles=("Cantidad de Cargos", "Porcentaje de Cargos"))

for i, jurisdiccion in enumerate(leyendas_rrhh):
    color = f'rgba({(100 + i*25) % 255}, {(150 + i*35) % 255}, {(200 + i*45) % 255}, 0.8)'
    color_map[jurisdiccion] = color

for jurisdiccion in leyendas_rrhh:
    jurisdiccion_data = rrhh_agrupados[rrhh_agrupados['JURISDICCION'] == jurisdiccion]
    fig_rrhh.add_trace(
        go.Bar(x=jurisdiccion_data['Año'], y=jurisdiccion_data['Cantidad Cargos'], name=jurisdiccion, legendgroup=jurisdiccion,
               showlegend=True, text=jurisdiccion_data['Cantidad Cargos'].apply(lambda x: f'{x:,.0f}'), 
               hovertext=jurisdiccion_data['JURISDICCION'], marker_color=color_map[jurisdiccion]),
        row=1, col=1
    )
    fig_rrhh.add_trace(
        go.Bar(x=jurisdiccion_data['Año'], y=jurisdiccion_data['Porcentaje'], name=jurisdiccion, legendgroup=jurisdiccion,
               showlegend=False, text=jurisdiccion_data['Porcentaje'].apply(lambda x: f'{x:.0f}%'), 
               hovertext=jurisdiccion_data['J
