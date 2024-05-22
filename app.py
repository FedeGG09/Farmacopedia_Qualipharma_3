import streamlit as st
import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import tempfile
import os
from document_analysis import (
    extraer_texto_pdf,
    extraer_texto_docx,
    leer_archivo_texto,
    encontrar_diferencias,
    vectorizar_y_tokenizar_diferencias,
    tokenizar_lineamientos,
    almacenar_reglas_vectorizadas,
    cargar_y_vectorizar_manual
)
from utils import verify_differences_compliance  # Importar la nueva función
from templatesStreamlit import css, user_template2, bot_template2

# Función para procesar documentos
def procesar_documentos(uploaded_reference_file, uploaded_compare_file, reference_file_type, compare_file_type):
    texto_referencia = extraer_texto(reference_file_type, uploaded_reference_file)
    texto_comparar = extraer_texto(compare_file_type, uploaded_compare_file)

    tokens_referencia = tokenizar_lineamientos([texto_referencia])
    diferencias = encontrar_diferencias(texto_comparar, texto_referencia)

    if diferencias:
        diferencias_vectorizadas = vectorizar_y_tokenizar_diferencias(
            diferencias, tokens_referencia, uploaded_compare_file.name, uploaded_reference_file.name
        )
        st.success("Las diferencias entre los documentos han sido encontradas y vectorizadas.")
        st.header("Diferencias Encontradas")
        diferencias_tabla = [
            [diferencia.get('seccion', 'N/A'), 
             diferencia.get('contenido_referencia', 'N/A'), 
             diferencia.get('contenido_documento', 'N/A'), 
             diferencia.get('tipo', 'N/A'),
             diferencia.get('recomendacion', 'N/A')]
            for diferencia in diferencias_vectorizadas
        ]
        st.table(pd.DataFrame(diferencias_tabla, columns=["Línea", "Sección", "Contenido de Referencia", "Tipo", "Recomendación"]))
    else:
        st.info("No se encontraron diferencias entre los documentos.")

# Función para extraer texto según el tipo de archivo
def extraer_texto(file_type, file):
    if file_type == "pdf":
        return extraer_texto_pdf(file)
    elif file_type == "docx":
        return extraer_texto_docx(file)
    elif file_type == "txt":
        return leer_archivo_texto(file)
    return ""

# Función para cargar y vectorizar el manual
def load_manual(texto_manual, indice_manual):
    tokens_referencia = tokenizar_lineamientos([texto_manual])
    almacenar_reglas_vectorizadas(texto_manual, tokens_referencia, indice_manual)
    st.session_state.tokens_referencia = tokens_referencia
    st.success("Manual cargado y vectorizado con éxito.")
    return tokens_referencia

# Función para verificar cumplimiento de archivo
def verify_file_compliance(tokens_referencia, texto_comparar):
    diferencias = encontrar_diferencias(texto_comparar, " ".join(tokens_referencia))
    if diferencias:
        st.warning("El documento no cumple con las normativas establecidas en el manual de referencia.")
        st.header("Diferencias Encontradas")
        diferencias_tabla = [
            [diferencia.get('seccion', 'N/A'), 
             diferencia.get('contenido_referencia', 'N/A'), 
             diferencia.get('contenido_documento', 'N/A'), 
             diferencia.get('tipo', 'N/A'),
             diferencia.get('recomendacion', 'N/A')]
            for diferencia in diferencias
        ]
        st.table(pd.DataFrame(diferencias_tabla, columns=["Línea", "Sección", "Contenido de Referencia", "Tipo", "Recomendación"]))
    else:
        st.success("El documento cumple con las normativas establecidas en el manual de referencia.")

# Función para comparar dos archivos adicionales con el manual
def compare_additional_files(tokens_referencia, file1, file2, file1_type, file2_type):
    texto1 = extraer_texto(file1_type, file1)
    texto2 = extraer_texto(file2_type, file2)

    diferencias1 = encontrar_diferencias(texto1, texto2)
    diferencias_vectorizadas1 = vectorizar_y_tokenizar_diferencias(
        diferencias1, tokens_referencia, file1.name, file2.name
    )
    
    diferencias2 = encontrar_diferencias(texto2, texto1)
    diferencias_vectorizadas2 = vectorizar_y_tokenizar_diferencias(
        diferencias2, tokens_referencia, file2.name, file1.name
    )

    st.success("Las diferencias entre los documentos adicionales han sido encontradas y vectorizadas.")
    st.header("Diferencias Encontradas entre los Documentos Adicionales")
    diferencias_tabla = [
        [diferencia.get('seccion', 'N/A'), 
         diferencia.get('contenido_referencia', 'N/A'), 
         diferencia.get('contenido_documento', 'N/A'), 
         diferencia.get('tipo', 'N/A'),
         diferencia.get('recomendacion', 'N/A')]
        for diferencia in diferencias_vectorizadas1 + diferencias_vectorizadas2
    ]
    st.table(pd.DataFrame(diferencias_tabla, columns=["Línea", "Sección", "Contenido de Referencia", "Tipo", "Recomendación"]))
    
    return diferencias_vectorizadas1, diferencias_vectorizadas2

# Función para leer los documentos y convertirlos en chunks
def load_documents(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
    return docs

def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = CTransformers(model="mistral-7b-instruct-v0.1.Q4_K_M.gguf", config={'max_new_tokens': 128, 'temperature': 0.01})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                               retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                               memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template2.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template2.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Interfaz Streamlit
st.set_page_config(page_title="Qualipharma - Analytics Town", page_icon="🧪")
st.title("Qualipharma - Analytics Town")

st.sidebar.image("https://raw.githubusercontent.com/FedeGG09/Qualipharma_2/main/data/input/cropped-qualipharma_isologo_print-e1590563965410-300x300.png", use_column_width=True)  # Añadir logo

# Sección para cargar el manual de referencia
st.sidebar.header("Cargar Manual de Referencia")
uploaded_reference_file = st.sidebar.file_uploader("Subir archivo de referencia", type=["pdf", "txt", "docx"])
if uploaded_reference_file:
    reference_file_type = uploaded_reference_file.name.split(".")[-1]
    st.sidebar.success(f"Archivo de referencia {uploaded_reference_file.name} cargado con éxito.")

# Sección para cargar el documento a comparar
st.sidebar.header("Cargar Documento a Comparar")
uploaded_compare_file = st.sidebar.file_uploader("Subir archivo a comparar", type=["pdf", "txt", "docx"])
if uploaded_compare_file:
    compare_file_type = uploaded_compare_file.name.split(".")[-1]
    st.sidebar.success(f"Archivo a comparar {uploaded_compare_file.name} cargado con éxito.")

# Botón para procesar documentos
if st.sidebar.button("Procesar Documentos") and uploaded_reference_file and uploaded_compare_file:
    procesar_documentos(uploaded_reference_file, uploaded_compare_file, reference_file_type, compare_file_type)

# Botón para cargar y vectorizar el manual
if st.sidebar.button("Cargar y Vectorizar Manual") and uploaded_reference_file:
    texto_manual = extraer_texto(reference_file_type, uploaded_reference_file)
    indice_manual = [
        "2.1. Minor variations of Type IA",
        "2.1.2. Type IA variations review for mutual recognition procedure",
        "2.1.3. Type IA variations review for purely national procedure",
        "2.1.4. Type IA variations review for centralised procedure",
        "2.2. Minor variations of Type IB",
        "2.2.1. Submission of Type IB notifications",
        "2.2.2. Type IB variations review for mutual recognition procedure",
        "2.2.3. Type IB variations review for purely national procedure",
        "2.2.4. Type IB variations review for centralised procedure",
        "2.3. Major variations of Type II",
        "2.3.1. Submission of Type II applications",
        "2.3.2. Type II variations assessment for mutual recognition procedure",
        "2.3.3. Outcome of Type II variations assessment for mutual recognition procedure",
        "2.3.4. Type II variations assessment for purely national procedure",
        "2.3.5. Outcome of Type II variations assessment for purely national procedure",
        "2.3.6. Type II variations assessment for centralised procedure",
        "2.3.7. Outcome of Type II variations assessment in centralised procedure",
        "2.4. Extensions",
        "2.4.1. Submission of Extensions applications",
        "2.4.2. Extension assessment for national procedure",
        "2.4.3. Extension assessment for centralised procedure"
    ]
    tokens_referencia = load_manual(texto_manual, indice_manual)
    st.session_state['manual_cargado'] = True
    st.session_state['texto_manual'] = texto_manual

# Sección para comparar documentos adicionales
st.sidebar.header("Comparar Documentos Adicionales")
uploaded_file1 = st.sidebar.file_uploader("Subir primer archivo adicional", type=["pdf", "txt", "docx"])
uploaded_file2 = st.sidebar.file_uploader("Subir segundo archivo adicional", type=["pdf", "txt", "docx"])
if uploaded_file1 and uploaded_file2:
    file1_type = uploaded_file1.name.split(".")[-1]
    file2_type = uploaded_file2.name.split(".")[-1]
    st.sidebar.success(f"Archivos adicionales {uploaded_file1.name} y {uploaded_file2.name} cargados con éxito.")

# Botón para comparar documentos adicionales
if st.sidebar.button("Comparar Documentos Adicionales") and uploaded_file1 and uploaded_file2:
    if st.session_state.get('manual_cargado', False):
        tokens_referencia = st.session_state.get('tokens_referencia', [])
        diferencias_vectorizadas1, diferencias_vectorizadas2 = compare_additional_files(tokens_referencia, uploaded_file1, uploaded_file2, file1_type, file2_type)
        st.session_state['diferencias_vectorizadas1'] = diferencias_vectorizadas1
        st.session_state['diferencias_vectorizadas2'] = diferencias_vectorizadas2
    else:
        st.error("Primero debes cargar y vectorizar el manual de referencia.")

# Botón para verificar cumplimiento de diferencias
if st.sidebar.button("Verificar Cumplimiento de Diferencias") and uploaded_file1 and uploaded_file2:
    if st.session_state.get('manual_cargado', False):
        tokens_referencia = st.session_state.get('tokens_referencia', [])
        diferencias_vectorizadas1 = st.session_state.get('diferencias_vectorizadas1', [])
        diferencias_vectorizadas2 = st.session_state.get('diferencias_vectorizadas2', [])

        verify_differences_compliance(diferencias_vectorizadas1, tokens_referencia)
        verify_differences_compliance(diferencias_vectorizadas2, tokens_referencia)
    else:
        st.error("Primero debes cargar y vectorizar el manual de referencia.")

# Inicializar estado de sesión para chat
if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Cargar archivos para el chat
st.header("Cargar Archivos para el Chat")
uploaded_files = st.file_uploader("Sube tus archivos (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    documents = load_documents(uploaded_files)
    text_chunks = split_text_into_chunks(documents)
    vectorstore = get_vectorstore(text_chunks)
    st.session_state.conversation = get_conversation_chain(vectorstore)
    st.success("Archivos cargados y procesados para el chat.")

# Interfaz de chat
st.header("Chat con el Asistente")
user_question = st.text_input("Haz una pregunta sobre los documentos:")
if user_question:
    handle_userinput(user_question)
