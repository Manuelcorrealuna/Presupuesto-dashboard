import streamlit as st
import pandas as pd
import plotly.express as px

# Cargar datos
gastos = pd.read_csv('data/Gastos.csv')
recursos_humanos = pd.read_csv('data/RecursosHumanos.csv')

st.title('Análisis de Presupuesto Argentino 2022-2025')

# Filtros
jurisdiccion = st.sidebar.selectbox('JURISDICCIÓN', gastos['JURISDICCIÓN'].unique())
subjurisdiccion = st.sidebar.selectbox('SUBJURISDICCIÓN', gastos['SUBJURISDICCIÓN'].unique())
anio = st.sidebar.selectbox('Año', ['2022', '2023', '2024', '2025'])

# Gráficos
gastos_filtrados = gastos[(gastos['JURISDICCIÓN'] == jurisdiccion) & 
                          (gastos['SUBJURISDICCIÓN'] == subjurisdiccion) & 
                          (gastos['Año'] == int(anio))]

fig_gastos = px.bar(gastos_filtrados, x='Objeto', y='Monto', title='Objeto del Gasto')
st.plotly_chart(fig_gastos)

# Aquí iría el gráfico de Recursos Humanos y la llamada a la IA

if st.button('Generar Análisis'):
    st.write("Aquí iría el análisis generado por la IA")