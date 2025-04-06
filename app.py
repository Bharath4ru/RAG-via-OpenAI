import os
import base64
import tempfile
import time
import uuid
import streamlit as st
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
import faiss
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core import Settings
from openai import OpenAI

# Load API keys
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Streamlit setup
st.set_page_config(page_title="🎤 Voice RAG Assistant", layout="centered")
st.title("🎤 Voice-Based RAG Chatbot")
st.markdown("---")

# Initialize session state variables if they don't exist
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = ""
if 'last_voice' not in st.session_state:
    st.session_state.last_voice = None
if 'last_tts_model' not in st.session_state:
    st.session_state.last_tts_model = None
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'response_audio' not in st.session_state:
    st.session_state.response_audio = None
if 'query_in_progress' not in st.session_state:
    st.session_state.query_in_progress = False
if 'last_query_source' not in st.session_state:
    st.session_state.last_query_source = None
if 'text_input_key' not in st.session_state:
    st.session_state.text_input_key = str(uuid.uuid4())
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = True

# Sidebar: Upload PDF and service selection
with st.sidebar:
    st.header("📄 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    st.header("🔧 Voice Settings")
    
    # Voice selection for OpenAI
    st.subheader("🎤 Voice Selection")
    
    # OpenAI voice options
    voice_options = [
        "alloy", "echo", "fable", "onyx", "nova", "shimmer"
    ]
    selected_voice = st.selectbox("Select OpenAI Voice", voice_options)
    
    # Add model selection for OpenAI TTS
    tts_models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
    selected_tts_model = st.selectbox("Select OpenAI TTS Model", tts_models)
    
    # Option to disable audio playback
    st.session_state.audio_enabled = st.checkbox("Enable audio playback", value=st.session_state.audio_enabled)
    
    st.markdown("---")

    # Check if configurations have changed and reset the query and answer
    if (st.session_state.last_voice != selected_voice or 
        st.session_state.last_tts_model != selected_tts_model):
        
        # Clear previous query and answer
        st.session_state.user_query = ""
        st.session_state.current_answer = ""
        st.session_state.response_audio = None
        
        # Update stored configuration
        st.session_state.last_voice = selected_voice
        st.session_state.last_tts_model = selected_tts_model

# Initialize index
def init_index(file_path):
    # Silence deprecation warnings with a context manager if needed
    llm = Gemini(model="models/gemini-2.0-flash-exp")
    embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    faiss_index = faiss.IndexFlatL2(768)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

# Generate TTS response using OpenAI and display as Streamlit audio
def stream_tts_response_openai(answer_text, voice, model):
    if not st.session_state.audio_enabled:
        st.info("Audio playback is disabled. Enable it in the sidebar if you want spoken responses.")
        return
        
    # Generate audio with minimal UI feedback
    with st.spinner("Generating audio response..."):
        try:
            # Generate speech with OpenAI TTS
            response = openai_client.audio.speech.create(
                model=model,
                voice=voice,
                input=answer_text,
                response_format="mp3"
            )
            
            # Store the audio in session state for persistence
            st.session_state.response_audio = response.content
            
        except Exception as e:
            st.error(f"Error with OpenAI TTS: {str(e)}")
            st.session_state.response_audio = None

# Transcribe audio using OpenAI
def transcribe_audio_openai(audio_data):
    # Create a unique temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data)
        tmp_path = tmp_file.name
    
    try:
        # Transcribe using OpenAI
        with open(tmp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                language="en"
            )
        return transcript.text
    except Exception as e:
        st.error(f"Error with OpenAI transcription: {str(e)}")
        return ""
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Clear the interaction
def clear_interaction():
    st.session_state.user_query = ""
    st.session_state.current_answer = ""
    st.session_state.audio_bytes = None
    st.session_state.response_audio = None
    st.session_state.query_in_progress = False
    # Generate a new key for the text input to force a refresh
    st.session_state.text_input_key = str(uuid.uuid4())

# Reset the text input field
def reset_text_input():
    st.session_state.text_input_key = str(uuid.uuid4())

# Main app logic
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    index = init_index(tmp_path)
    query_engine = index.as_query_engine()
    
    # Display current query and answer if they exist
    if st.session_state.user_query:
        st.subheader("Your Question:")
        st.info(st.session_state.user_query)
    
    if st.session_state.current_answer:
        st.subheader("Answer:")
        st.success(st.session_state.current_answer)
        
        # Display the audio player if there's audio response
        if st.session_state.response_audio and st.session_state.audio_enabled:
            audio_container = st.container()
            with audio_container:
                # Display speaker gif if available
                gif_path = "speaker.gif"
                if os.path.exists(gif_path):
                    with open(gif_path, "rb") as f:
                        data_url = base64.b64encode(f.read()).decode("utf-8")
                        st.markdown(f'<img src="data:image/gif;base64,{data_url}" style="height:100px;">', unsafe_allow_html=True)
                
                # Use Streamlit's native audio player with persistent audio
                st.audio(st.session_state.response_audio, format="audio/mp3", start_time=0)
    
    # Create a horizontal layout for input elements
    col1, col2 = st.columns([1, 4])
    
    # Audio recorder in the first (smaller) column
    with col1:
        audio_bytes = audio_recorder(recording_color="#f10c49", neutral_color="#6aa36f", 
                                    key="audio_recorder", icon_size="2x")
        
        if audio_bytes and audio_bytes != st.session_state.audio_bytes:
            # Clear previous interaction first
            clear_interaction()
            
            st.session_state.audio_bytes = audio_bytes
            st.session_state.query_in_progress = True
            st.session_state.last_query_source = "voice"
            
            # Transcribe with minimal UI feedback
            with st.spinner("Transcribing your voice..."):
                # Use OpenAI for STT
                transcribed_text = transcribe_audio_openai(audio_bytes)
                
            if transcribed_text:
                st.session_state.user_query = transcribed_text
                st.rerun()
            else:
                st.error("Failed to transcribe audio. Please try again.")
                st.session_state.query_in_progress = False
    
    # Text input in the second (larger) column
    with col2:
        # Use dynamic key to force reset
        user_query = st.text_input("Ask a question about your document:", 
                                  key=st.session_state.text_input_key)
        
        # Process text input if entered
        if user_query:
            # Only process if this is a new query (not the one we're already processing)
            if (st.session_state.user_query != user_query or 
                st.session_state.last_query_source != "text"):
                
                # Clear previous interaction
                st.session_state.user_query = user_query
                st.session_state.current_answer = ""
                st.session_state.response_audio = None
                st.session_state.query_in_progress = True
                st.session_state.last_query_source = "text"
                
                # Rerun to process the new query
                st.rerun()
    
    # Process the query if one exists and is not already in progress
    if st.session_state.user_query and st.session_state.query_in_progress:
        # Process with Gemini - minimal UI feedback
        with st.spinner("Processing your question..."):
            response = query_engine.query(st.session_state.user_query)
            answer = response.response
            
            # Store the current answer
            st.session_state.current_answer = answer
            st.session_state.query_in_progress = False
            
            # Generate audio response (this now stores in session state)
            stream_tts_response_openai(answer, selected_voice, selected_tts_model)
            
            # Display the answer
            st.subheader("Answer:")
            st.success(answer)
            
            # Display audio player (will be handled in the main flow after rerun)
            st.rerun()
            
            # Reset text input field to prepare for next query
            reset_text_input()
            
    # Clean up temp file after app is done
    try:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    except Exception:
        pass
else:
    st.warning("📄 Please upload a PDF from the sidebar to begin.")
