# 🎤 Voice-Based RAG Chatbot

A Streamlit application that combines voice interaction with Retrieval-Augmented Generation (RAG) to provide an intuitive way to query PDF documents using either voice or text inputs.

## 📋 Features

- **Voice & Text Interaction**: Ask questions about your documents using either voice commands or text input
- **PDF Document Processing**: Upload any PDF document and query its contents
- **Text-to-Speech Response**: Get audio responses using OpenAI's text-to-speech technology
- **Multiple Voice Options**: Choose from a variety of OpenAI TTS voices
- **TTS Model Selection**: Select between different OpenAI TTS models for quality/speed tradeoffs
- **Advanced RAG Implementation**: Uses Gemini models and FAISS vector database for efficient document retrieval

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Speech-to-Text**: OpenAI Whisper (via API)
- **Text-to-Speech**: OpenAI TTS API
- **RAG Components**:
  - **LLM**: Google Gemini 2.0 Flash
  - **Embeddings**: Google Text Embedding 004
  - **Vector Store**: FAISS
  - **Document Processing**: LlamaIndex

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/voice-rag-assistant.git
   cd voice-rag-assistant
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## 📦 Dependencies

- streamlit
- dotenv
- audio_recorder_streamlit
- faiss-cpu
- llama-index
- openai

## 🏃‍♂️ Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Upload a PDF document using the sidebar uploader.

3. Configure voice settings in the sidebar:
   - Select an OpenAI voice (alloy, echo, fable, onyx, nova, shimmer)
   - Choose a TTS model (tts-1, tts-1-hd, gpt-4o-mini-tts)
   - Enable/disable audio playback

4. Ask questions about your document by:
   - Clicking the record button and speaking your question
   - Typing your question in the text input field

5. View the text response and listen to the audio answer.

## 🔄 Workflow

1. **Document Upload**: The app processes your PDF and creates embeddings using Gemini
2. **Query Input**: Provide your question via voice or text
3. **RAG Processing**: The system retrieves relevant document sections and generates an answer
4. **Response Generation**: The answer is displayed as text and converted to speech

## ⚙️ Configuration Options

| Setting | Options | Description |
|---------|---------|-------------|
| Voice | alloy, echo, fable, onyx, nova, shimmer | Different voice characters for TTS |
| TTS Model | tts-1, tts-1-hd, gpt-4o-mini-tts | Models with different quality/speed tradeoffs |
| Audio Playback | Enabled/Disabled | Option to turn off audio responses |

## 🧩 System Architecture

The application follows this process flow:
1. PDF parsing and embedding generation using Gemini models
2. Storage of embeddings in a FAISS vector database
3. Query processing via the LlamaIndex query engine
4. Speech-to-text conversion for voice inputs via OpenAI
5. Text-to-speech conversion for responses via OpenAI

## 📝 Notes

- The application requires an internet connection to access OpenAI and Google APIs.
- For optimal voice recording, use a good microphone in a quiet environment.
- Large PDF files may take longer to process initially.

## 🔮 Future Improvements

- Add support for more document formats (DOCX, TXT, etc.)
- Implement multi-document querying
- Add conversation history for follow-up questions
- Support for multiple languages
- Optimize for mobile usage

## 📄 License

[MIT License](LICENSE)

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
