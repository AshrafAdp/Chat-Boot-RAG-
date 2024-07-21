import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import speech_recognition as sr
from gtts import gTTS
import os

try:
    import pyaudio
except ImportError:
    st.error("PyAudio n'est pas installé. Veuillez l'installer en utilisant 'pip install pyaudio'.")

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Parlez maintenant...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio, language="fr-FR")
        st.write(f"Vous avez dit: {query}")
        return query
    except sr.UnknownValueError:
        st.write("Désolé, je n'ai pas compris.")
        return ""
    except sr.RequestError:
        st.write("Erreur de service de reconnaissance vocale.")
        return ""

def text_to_speech(text):
    tts = gTTS(text=text, lang='fr')
    audio_file = "response.mp3"
    tts.save(audio_file)
    return audio_file

def main():
    st.set_page_config(layout="wide")
    st.subheader("Retrieval Augmented Generation", divider="rainbow")
    
    # Initialiser les variables de session
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    if 'recognized_text' not in st.session_state:
        st.session_state.recognized_text = ""
    if 'recorded_audio' not in st.session_state:
        st.session_state.recorded_audio = None

    with st.sidebar:
        st.sidebar.title("Data Loader")
        pdf_docs = st.file_uploader("Upload Your PDFs", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Loading..."):
                pdf_content = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        pdf_content += page.extract_text()
                text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
                chunks = text_splitter.split_text(pdf_content)
                st.write(chunks)
                # OPEN IA API
                OPEN_API_KEY = "" 
             
                openai_embeddings = OpenAIEmbeddings(api_key=OPEN_API_KEY)
                openai_vector_store = FAISS.from_texts(texts=chunks, embedding=openai_embeddings)
                llm = ChatOpenAI(api_key=OPEN_API_KEY, model="gpt-4o")
                prompt = ChatPromptTemplate.from_template(
                    """
                    Answer the following question based only on the provided context:
                    <context>
                      {context}
                    </context>
                    Question: {input}
                    """
                )
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = openai_vector_store.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                st.session_state.retrieve_chain = retrieval_chain

    st.subheader("Chatbot zone")
    
    # Gestion de l'enregistrement vocal
    if st.session_state.recording:
        st.write("Enregistrement en cours... Parlez maintenant.")
    
    if st.button("Commencer l'enregistrement"):
        st.session_state.recording = True
        st.session_state.user_question = ""
    
    if st.session_state.recording and st.button("Arrêter l'enregistrement"):
        st.session_state.recognized_text = recognize_speech()
        st.session_state.recording = False
        st.session_state.user_question = st.session_state.recognized_text
        st.experimental_rerun()
    
    if st.session_state.user_question:
        user_question = st.session_state.user_question
        response = st.session_state.retrieve_chain.invoke({"input": user_question})
        st.markdown(response["answer"], unsafe_allow_html=True)
        audio_file = text_to_speech(response["answer"])
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")
        st.session_state.user_question = ""
        st.session_state.recognized_text = ""

    user_question = st.text_input("Ask your question :")
    if user_question:
        response = st.session_state.retrieve_chain.invoke({"input": user_question})
        st.markdown(response["answer"], unsafe_allow_html=True)
        audio_file = text_to_speech(response["answer"])
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")

if __name__ == "__main__":
    main()
