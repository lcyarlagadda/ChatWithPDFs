# PDF Chat System

A modern web application that allows you to chat with your PDF documents using AI. Built with React/Next.js frontend and FastAPI backend.

## Features

- **PDF Upload**: Upload multiple PDF files for processing
- **AI Chat**: Ask questions about your documents using Mistral-7B
- **Hybrid Retrieval**: Combines dense and sparse retrieval for better results
- **Real-time Metrics**: View processing times and token usage
- **Modern UI**: Clean, responsive interface built with Tailwind CSS
- **Streaming Responses**: Real-time answer generation
- **Mobile Friendly**: Works on desktop and mobile devices

## Architecture

- **Frontend**: Next.js React application with TypeScript
- **Backend**: FastAPI server with Python
- **AI Models**: 
  - Mistral-7B-Instruct for text generation
  - all-MiniLM-L6-v2 for embeddings
  - CrossEncoder for reranking
- **Vector Store**: FAISS for similarity search
- **Text Processing**: PyMuPDF for PDF extraction

## Quick Start

### Backend (Google Colab)

1. Open the [Colab notebook](backend/colab_deployment.ipynb)
2. Enable GPU: Runtime → Change runtime type → GPU
3. Set your Hugging Face token
4. Run all cells
5. Copy the ngrok URL

### Frontend (Local)

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Set environment variable:
```bash
echo "NEXT_PUBLIC_API_URL=https://your-ngrok-url.ngrok.io" > .env.local
```

4. Run development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000)

## Project Structure

```
├── frontend/                 # Next.js React application
│   ├── app/
│   │   ├── components/      # React components
│   │   ├── services/        # API service layer
│   │   ├── globals.css      # Global styles
│   │   ├── layout.tsx       # Root layout
│   │   └── page.tsx         # Main page
│   ├── package.json
│   ├── tailwind.config.js
│   └── next.config.js
├── backend/                 # FastAPI server
│   ├── main.py             # FastAPI app and endpoints
│   ├── model_manager.py    # Model loading and caching
│   ├── pdf_processor.py    # PDF text extraction
│   ├── retriever.py        # Hybrid retrieval system
│   ├── rag_system.py       # RAG pipeline
│   ├── requirements.txt    # Python dependencies
│   └── colab_deployment.ipynb  # Colab deployment notebook
├── DEPLOYMENT.md           # Detailed deployment instructions
└── README.md              # This file
```

## API Endpoints

### Health Check
```
GET /health
```

### Upload PDFs
```
POST /upload-pdfs
Content-Type: multipart/form-data
```

### Ask Question
```
POST /ask-question
Content-Type: application/json
{
  "question": "Your question here",
  "settings": {
    "numChunks": 3,
    "chunkSize": 500,
    "temperature": 0.2,
    "maxTokens": 512
  }
}
```

### Get Status
```
GET /status
```

## Configuration

### Environment Variables

**Backend:**
- `HUGGINGFACE_HUB_TOKEN`: Your Hugging Face API token

**Frontend:**
- `NEXT_PUBLIC_API_URL`: Backend API URL (e.g., ngrok URL)

### Settings

The application supports various settings:

- **Number of Chunks**: How many text chunks to retrieve (1-10)
- **Chunk Size**: Size of text chunks in tokens (200-1000)
- **Temperature**: Controls randomness in responses (0.1-1.0)
- **Max Tokens**: Maximum tokens in response (100-1000)
- **Model Type**: Choose between Mistral-7B, FLAN-T5, GPT-2
- **Retriever Type**: Hybrid, Dense only, or Sparse only
- **Noise Filtering**: Enable/disable PDF noise removal
- **Debug Mode**: Show retrieved chunks and metrics

## Deployment Options

### Backend Deployment

1. **Google Colab** (Recommended for free GPU)
   - Use the provided notebook
   - Free GPU access
   - Easy setup

2. **Local Server**
   - Install Python dependencies
   - Run `python main.py`
   - Requires GPU for good performance

3. **Cloud Platforms**
   - Google Cloud Run
   - AWS Lambda
   - Azure Functions

### Frontend Deployment

1. **Vercel** (Recommended)
   - Connect GitHub repository
   - Automatic deployments
   - Free tier available

2. **Netlify**
   - Drag and drop deployment
   - Free tier available

3. **Local Development**
   - Run `npm run dev`
   - Access at localhost:3000

## Troubleshooting

### Common Issues

1. **Hugging Face Token Error**
   - Get token from https://huggingface.co/settings/tokens
   - Set `HUGGINGFACE_HUB_TOKEN` environment variable

2. **CUDA/GPU Issues**
   - Ensure GPU is enabled in Colab
   - Check GPU availability: `!nvidia-smi`

3. **Memory Issues**
   - Use smaller models
   - Reduce batch sizes
   - Monitor RAM usage

4. **CORS Issues**
   - Backend has CORS enabled for all origins
   - Check if frontend URL is correct

5. **ngrok Issues**
   - Free ngrok URLs change on restart
   - Consider paid ngrok for stable URLs

### Performance Tips

1. **Model Loading**
   - Models are cached after first load
   - Restart kernel to clear cache if needed

2. **PDF Processing**
   - Large PDFs may take time to process
   - Consider splitting large documents

3. **Memory Management**
   - Monitor Colab's RAM usage
   - Restart runtime if memory is full

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [Mistral AI](https://mistral.ai/) for the language model
- [Hugging Face](https://huggingface.co/) for model hosting
- [Google Colab](https://colab.research.google.com/) for free GPU access
- [Next.js](https://nextjs.org/) and [FastAPI](https://fastapi.tiangolo.com/) for the frameworks


