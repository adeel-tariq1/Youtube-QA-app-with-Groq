import streamlit as st
import yt_dlp
import whisper
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
import os
import torch
from groq import Groq
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import tempfile

# Custom Embeddings class for SentenceTransformer
@st.cache_resource
def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = load_model(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text]).tolist()[0]

# Set the title of the Streamlit app
st.title("YouTube Q&A and Summarization App with Groq")
st.markdown("""
    #### Description
    Upload a YouTube video and get a Q&A Assistant or a Summary using Groq's fast LLMs.
    """)

# Sidebar for API key and options
st.sidebar.subheader("Configuration")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

st.sidebar.subheader("About the App")
st.sidebar.info("This App has been developed using Whisper and Langchain with Groq LLMs. For Q&A it uses llama-3.1-8b-instant (fast and efficient)")

st.sidebar.subheader("Select an Option")
selected_option = st.sidebar.selectbox("Please select an option", ["Q&A", "Summarize"])

# Get the YouTube URL from the user
url = st.text_input("Add the URL of the YouTube Video:")

# Create or retrieve session state variables
if "model" not in st.session_state:
    st.session_state.model = None
if "url" not in st.session_state:
    st.session_state.url = ""
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "groq_client" not in st.session_state:
    st.session_state.groq_client = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed" not in st.session_state:
    st.session_state.processed = False

# Initialize Groq client if API key is provided
if groq_api_key and not st.session_state.groq_client:
    try:
        st.session_state.groq_client = Groq(api_key=groq_api_key)
        st.sidebar.success("✅ Groq client initialized successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to initialize Groq client: {e}")

# Detect the appropriate device for Whisper
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the base Whisper model (cached)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base", device=DEVICE)

whisper_model = load_whisper_model()

import yt_dlp

import os
import tempfile
import shutil
from typing import Any

def download_youtube_audio(youtube_url):
    """Download YouTube audio and ensure proper file extension"""
    try:
        # Create a specific temporary file with known path
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio")
        
        st.info("📥 Downloading YouTube audio...")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
            # Get video info first
            info = ydl.extract_info(youtube_url, download=False)
            if not info:
                st.error("❌ Invalid YouTube URL")
                return None
            
            st.info(f"🎬 Video: {info.get('title', 'Unknown')}")
            
            # Download the audio
            ydl.download([youtube_url])
        
        # Debug: List all files in temp directory
        all_files = os.listdir(temp_dir)
        st.info(f"📁 Files downloaded: {all_files}")
        
        # Find the actual downloaded file
        actual_audio_path = None
        for file in all_files:
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 1000:
                actual_audio_path = file_path
                break
        
        if actual_audio_path:
            file_size = os.path.getsize(actual_audio_path) // 1024
            st.success(f"✅ Downloaded: {os.path.basename(actual_audio_path)} ({file_size} KB)")
            
            # **CRITICAL FIX: Add proper file extension**
            final_audio_path = add_audio_extension(actual_audio_path)
            return final_audio_path
        else:
            st.error("❌ No valid audio file found after download")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
            
    except Exception as e:
        st.error(f"❌ Download error: {str(e)}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None

def add_audio_extension(file_path):
    """Detect audio format and add correct file extension"""
    try:
        import magic  # python-magic library for file type detection
        
        # Detect the actual file type
        file_type = magic.from_file(file_path, mime=True)
        st.info(f"🔍 Detected file type: {file_type}")
        
        new_path = file_path
        
        if 'webm' in file_type:
            new_path = file_path + '.webm'
        elif 'mp4' in file_type or 'm4a' in file_type:
            new_path = file_path + '.m4a'
        elif 'mp3' in file_type:
            new_path = file_path + '.mp3'
        elif 'wav' in file_type:
            new_path = file_path + '.wav'
        else:
            # Default to .webm (common YouTube audio format)
            new_path = file_path + '.webm'
        
        # Rename the file with proper extension
        if new_path != file_path:
            os.rename(file_path, new_path)
            st.info(f"📝 Renamed to: {os.path.basename(new_path)}")
        
        return new_path
        
    except ImportError:
        # If magic is not available, use simple extension detection
        st.warning("⚠️ python-magic not installed. Using fallback method.")
        return add_audio_extension_fallback(file_path)

def add_audio_extension_fallback(file_path):
    """Fallback method without magic library"""
    try:
        # Read first few bytes to detect format
        with open(file_path, 'rb') as f:
            header = f.read(20)  # Read first 20 bytes
        
        # Common audio file signatures
        if header.startswith(b'RIFF'):
            new_path = file_path + '.wav'
        elif header.startswith(b'ID3') or header.startswith(b'\xFF\xFB'):
            new_path = file_path + '.mp3'
        elif header.startswith(b'\x1A\x45\xDF\xA3'):  # WebM signature
            new_path = file_path + '.webm'
        elif header.startswith(b'ftyp'):  # MP4 signature
            new_path = file_path + '.m4a'
        else:
            # Try common extensions
            for ext in ['.webm', '.m4a', '.mp3', '.wav']:
                test_path = file_path + ext
                try:
                    # Test if Whisper can read it
                    import whisper
                    model = whisper.load_model("base")
                    result = model.transcribe(test_path, word_timestamps=False)
                    if result.get("text"):
                        new_path = test_path
                        st.info(f"✅ Found working format: {ext}")
                        break
                except:
                    continue
            else:
                new_path = file_path + '.webm'  # Default fallback
        
        if new_path != file_path:
            os.rename(file_path, new_path)
            st.info(f"📝 Renamed to: {os.path.basename(new_path)}")
        
        return new_path
        
    except Exception as e:
        st.warning(f"⚠️ Could not detect file format: {e}")
        # Default to webm
        new_path = file_path + '.webm'
        os.rename(file_path, new_path)
        return new_path

def transcribe_with_groq(audio_path):
    if not st.session_state.groq_client:
        st.error("Groq client not initialized")
        return ""
    with open(audio_path, "rb") as f:
        result = st.session_state.groq_client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3"
        )
    return result.text

def convert_to_wav(audio_path):
    """Convert audio file to WAV format using ffmpeg"""
    try:
        import subprocess
        import sys
        
        wav_path = audio_path + '.wav'
        
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except:
            st.error("❌ FFmpeg not found. Please install ffmpeg.")
            return None
        
        # Convert to WAV
        cmd = [
            'ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le', 
            '-ac', '1', '-ar', '16000', '-y', wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(wav_path):
            st.info("✅ Converted to WAV format")
            return wav_path
        else:
            st.error(f"❌ Conversion failed: {result.stderr}")
            return None
            
    except Exception as e:
        st.error(f"❌ Conversion error: {e}")
        return None

def groq_summarize_text(text, max_length=200):
    """Use Groq for summarization"""
    try:
        if not st.session_state.groq_client:
            st.error("Groq client not initialized")
            return ""
        
        # Truncate very long text to avoid token limits
        if len(text) > 8000:
            text = text[:8000] + "... [text truncated]"
        
        prompt = f"Please provide a concise summary of the following text in 2-3 sentences:\n\n{text}"
        
        response = st.session_state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_length,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return ""

# Main processing logic
if selected_option == "Q&A":
    if st.button("🚀 Extract Audio and Transcribe"):
        if url and groq_api_key:
            with st.spinner("Processing YouTube video..."):
                # Validate URL first
                if not url.startswith(('http://', 'https://')):
                    st.error("❌ Please enter a valid URL starting with http:// or https://")
                    st.stop()
                
                # Download audio
                audio_path = download_youtube_audio(url)
                
                if audio_path:
                    st.info(f"🔍 Audio file path: {audio_path}")
                    
                    # Transcribe audio
                    transcription = transcribe_with_groq(audio_path)
                    
                    if transcription and len(transcription.strip()) > 10:
                        st.session_state.transcription = transcription
                        
                        try:
                            # Initialize embeddings
                            embeddings = SentenceTransformerEmbeddings()
                            
                            # Split transcription into chunks
                            text_chunks = []
                            chunk_size = 1000
                            for i in range(0, len(transcription), chunk_size):
                                chunk = transcription[i:i + chunk_size].strip()
                                if chunk and len(chunk) > 10:
                                    text_chunks.append(chunk)
                            
                            if not text_chunks:
                                st.error("❌ No valid text chunks created")
                                # Clean up audio file
                                if os.path.exists(audio_path):
                                    os.unlink(audio_path)
                                st.stop()
                            
                            # Create vector store
                            vStore = Chroma.from_texts(
                                text_chunks, 
                                embeddings, 
                                metadatas=[{"source": f"chunk_{i+1}"} for i in range(len(text_chunks))]
                            )
                            
                            # Create retriever
                            retriever = vStore.as_retriever(search_kwargs={'k': 3})
                            
                            # Initialize Groq LLM
                            from pydantic import SecretStr
                            llm = ChatGroq(
                                api_key=SecretStr(groq_api_key),
                                model="llama-3.1-8b-instant",
                                temperature=0.1
                            )
                            
                            # Create QA chain
                            qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                                llm=llm, 
                                chain_type="stuff", 
                                retriever=retriever
                            )
                            
                            st.session_state.model = qa_chain
                            st.session_state.vector_store = vStore
                            st.session_state.url = url
                            st.session_state.processed = True
                            st.success("✅ Q&A system ready! You can now ask questions.")
                            
                        except Exception as e:
                            st.error(f"❌ Error setting up Q&A system: {e}")
                    
                    else:
                        st.error("❌ Transcription failed or too short")
                    
                    # Clean up audio file
                    try:
                        if audio_path and os.path.exists(audio_path):
                            # If it's in a temp directory, remove the entire directory
                            if 'temp' in audio_path:
                                import shutil
                                dir_path = os.path.dirname(audio_path)
                                if os.path.exists(dir_path):
                                    shutil.rmtree(dir_path, ignore_errors=True)
                            else:
                                os.unlink(audio_path)
                    except Exception as e:
                        st.warning(f"⚠️ Could not clean up files: {e}")
                else:
                    st.error("❌ Failed to download YouTube audio")
        else:
            if not url:
                st.warning("Please enter a YouTube video URL.")
            if not groq_api_key:
                st.warning("Please enter your Groq API key.")

    # Q&A Interface
    if st.session_state.processed and st.session_state.model and st.session_state.url:
        st.subheader("💡 Ask Questions about the Video")
        question = st.text_input("Enter your question:", placeholder="What is this video about?")
        
        if question and st.button("Get Answer"):
            with st.spinner("🔍 Searching for answer..."):
                try:
                    result = st.session_state.model({"question": question}, return_only_outputs=True)
                    
                    st.subheader("📖 Answer:")
                    st.write(result["answer"])
                    
                    with st.expander("🔍 View transcription context"):
                        st.text_area("Full Transcription", st.session_state.transcription, height=200, key="transcription_display")
                        
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

elif selected_option == "Summarize":
    if st.button(" Summarize Video"):
        if url and groq_api_key:
            with st.spinner("Processing video for summarization..."):
                # Download audio
                audio_path = download_youtube_audio(url)
                
                if audio_path:
                    # Transcribe audio
                    transcription = transcribe_with_groq(audio_path)
                    
                    if transcription and len(transcription.strip()) > 50:
                        st.session_state.transcription = transcription
                        st.success(" Transcription completed!")
                        
                        # Display transcription
                        with st.expander("View Full Transcription"):
                            st.text_area("Transcription", transcription, height=200, key="summary_transcription")
                        
                        # Generate summary using Groq
                        with st.spinner("Generating summary..."):
                            summary = groq_summarize_text(transcription, max_length=200)
                            
                            if summary:
                                st.subheader("Video Summary:")
                                st.write(summary)
                            else:
                                st.error("Failed to generate summary")
                    
                    else:
                        st.error("Transcription failed or too short. Please try a different video.")
                    
                    # Clean up audio file
                    try:
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)
                    except Exception as e:
                        st.warning(f"Could not clean up audio file: {e}")
                else:
                    st.error("Failed to process YouTube video")
        else:
            if not url:
                st.warning("Please enter a YouTube video URL.")
            if not groq_api_key:
                st.warning("Please enter your Groq API key.")

# Additional features
from pytube import YouTube  # <-- Add this import for YouTube class

st.sidebar.markdown("---")
st.sidebar.subheader(" Video Info")
if st.session_state.url:
    try:
        yt = YouTube(st.session_state.url)
        st.sidebar.write(f"**Title:** {yt.title}")
        st.sidebar.write(f"**Duration:** {yt.length // 60} minutes")
        st.sidebar.write(f"**Views:** {yt.views:,}")
    except:
        pass

# Reset button
if st.sidebar.button("🔄 Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Requirements information
st.sidebar.markdown("---")
st.sidebar.info("""
**Models Used:**
- Whisper (Base) - Audio transcription
- SentenceTransformer - Text embeddings  
- Groq LLama 3.1 8B - Q&A and Summarization
""")