from PyPDF2 import PdfFileReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Función para extraer texto de archivos PDF
def extraer_texto_pdf(file):
    pdf = PdfFileReader(file)
    texto = ""
    for page in range(pdf.getNumPages()):
        texto += pdf.getPage(page).extractText()
    return texto

# Función para extraer texto de archivos DOCX
def extraer_texto_docx(file):
    doc = Document(file)
    texto = ""
    for para in doc.paragraphs:
        texto += para.text + "\n"
    return texto

# Función para leer archivos de texto plano
def leer_archivo_texto(file):
    return file.read().decode("utf-8")

# Función para encontrar diferencias entre dos textos
def encontrar_diferencias(texto_comparar, texto_referencia):
    diferencias = []
    lineas_comparar = texto_comparar.splitlines()
    lineas_referencia = texto_referencia.splitlines()

    for i, linea in enumerate(lineas_comparar):
        if i < len(lineas_referencia):
            if linea != lineas_referencia[i]:
                diferencias.append({
                    'linea': i + 1,
                    'contenido_referencia': lineas_referencia[i],
                    'contenido_documento': linea,
                    'tipo': 'Diferencia'
                })
        else:
            diferencias.append({
                'linea': i + 1,
                'contenido_referencia': '',
                'contenido_documento': linea,
                'tipo': 'Adicional'
            })

    for i in range(len(lineas_comparar), len(lineas_referencia)):
        diferencias.append({
            'linea': i + 1,
            'contenido_referencia': lineas_referencia[i],
            'contenido_documento': '',
            'tipo': 'Faltante'
        })

    return diferencias

# Función para tokenizar lineamientos
def tokenizar_lineamientos(textos):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(textos)
    return vectorizer.get_feature_names_out()

# Función para vectorizar y tokenizar diferencias
def vectorizar_y_tokenizar_diferencias(diferencias, tokens_referencia, nombre_documento, nombre_referencia):
    vectorizer = TfidfVectorizer()
    all_texts = [d['contenido_documento'] for d in diferencias] + [d['contenido_referencia'] for d in diferencias]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    diferencias_vectorizadas = []
    for diferencia in diferencias:
        doc_vector = vectorizer.transform([diferencia['contenido_documento']])
        ref_vector = vectorizer.transform([diferencia['contenido_referencia']])
        similitud = cosine_similarity(doc_vector, ref_vector)[0][0]

        diferencias_vectorizadas.append({
            'linea': diferencia['linea'],
            'seccion': diferencia.get('seccion', 'N/A'),
            'contenido_referencia': diferencia['contenido_referencia'],
            'contenido_documento': diferencia['contenido_documento'],
            'tipo': diferencia['tipo'],
            'similitud': similitud,
            'recomendacion': 'Revisar' if similitud < 0.8 else 'Ok'
        })

    return diferencias_vectorizadas

# Función para almacenar reglas vectorizadas
def almacenar_reglas_vectorizadas(texto_manual, tokens_referencia, indice_manual):
    # Implementar almacenamiento si es necesario
    pass

# Función para cargar y vectorizar el manual
def cargar_y_vectorizar_manual(file, file_type):
    texto_manual = extraer_texto(file_type, file)
    tokens_referencia = tokenizar_lineamientos([texto_manual])
    almacenar_reglas_vectorizadas(texto_manual, tokens_referencia, None)
    return tokens_referencia

# Función para verificar el cumplimiento de las diferencias
def verify_differences_compliance(diferencias, tokens_referencia):
    for diferencia in diferencias:
        doc_vector = vectorizer.transform([diferencia['contenido_documento']])
        ref_vector = vectorizer.transform([diferencia['contenido_referencia']])
        similitud = cosine_similarity(doc_vector, ref_vector)[0][0]

        diferencia['similitud'] = similitud
        diferencia['recomendacion'] = 'Revisar' if similitud < 0.8 else 'Ok'

# Esta función ahora está en utils.py, pero aquí está para referencia
def verificar_cumplimiento(texto_comparar, tokens_referencia):
    diferencias = encontrar_diferencias(texto_comparar, " ".join(tokens_referencia))
    diferencias_vectorizadas = vectorizar_y_tokenizar_diferencias(diferencias, tokens_referencia, "Comparar", "Referencia")
    return diferencias_vectorizadas
