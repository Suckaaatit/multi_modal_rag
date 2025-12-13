# Multi-Modal RAG System for Financial Documents

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed for financial documents containing text, tables, and charts.

<img width="896" height="380" alt="image" src="https://github.com/user-attachments/assets/8a79d30d-754e-4b99-ad43-2896f4c72749" />

<img width="900" height="437" alt="image" src="https://github.com/user-attachments/assets/41dbbe88-5b83-43e2-b241-fca8e5a353bc" />




## Features

- **Multi-Modal Ingestion**: Processes both text and images from PDFs
- **Vision AI Integration**: Uses Google's Gemini Vision to analyze and describe charts and figures
- **Vector Search**: Implements semantic search using ChromaDB
- **Citation & Source Attribution**: Provides page references for all information
- **Streamlit Web Interface**: User-friendly interface for document upload and querying

## Setup Instructions

1. **Prerequisites**:
   - Python 3.10 or higher
   - Google Gemini API Key (get it from [aistudio.google.com](https://aistudio.google.com/))

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. **Usage**:
   - Enter your Google API Key in the sidebar
   - Upload a PDF document
   - Ask questions about the document in the chat interface

## Technical Details

### System Architecture

1. **Ingestion Pipeline**:
   - Extracts text and images from PDFs using PyMuPDF
   - Processes images through Gemini Vision AI to generate text descriptions
   - Creates vector embeddings of all content

2. **Vector Database**:
   - Uses ChromaDB for efficient vector similarity search
   - Implements smart chunking for optimal retrieval

3. **Query Processing**:
   - Semantic search to find relevant document chunks
   - Context-aware response generation with citations

## File Structure

- `app.py`: Main Streamlit application
- `ingestion.py`: Handles PDF processing and image analysis
- `rag_engine.py`: Core RAG functionality and vector database operations
- `requirements.txt`: Python dependencies
- `README.md`: This documentation file

## Notes

- The system is designed to work with financial reports and similar structured documents
- For best results, use documents with clear text and well-defined charts/tables
- Processing time depends on document length and number of images
