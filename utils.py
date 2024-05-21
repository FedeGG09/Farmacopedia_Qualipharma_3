# utils.py
import streamlit as st
import pandas as pd

def verify_differences_compliance(diferencias_vectorizadas, tokens_referencia):
    diferencias_no_cumplen = []
    for diferencia in diferencias_vectorizadas:
        if diferencia['contenido_documento'] not in tokens_referencia:
            diferencias_no_cumplen.append(diferencia)

    if diferencias_no_cumplen:
        st.warning("Algunas diferencias no cumplen con las normativas establecidas en el manual de referencia.")
        st.header("Diferencias No Cumplen con el Manual")
        diferencias_tabla = [
            [diferencia.get('seccion', 'N/A'), 
             diferencia.get('contenido_referencia', 'N/A'), 
             diferencia.get('contenido_documento', 'N/A'), 
             diferencia.get('tipo', 'N/A'),
             diferencia.get('recomendacion', 'N/A')]
            for diferencia in diferencias_no_cumplen
        ]
        st.table(pd.DataFrame(diferencias_tabla, columns=["Línea", "Sección", "Contenido de Referencia", "Tipo", "Recomendación"]))
    else:
        st.success("Todas las diferencias cumplen con las normativas establecidas en el manual de referencia.")
