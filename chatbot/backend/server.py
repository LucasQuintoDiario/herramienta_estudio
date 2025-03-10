from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from typing import List, Dict, Optional
import cohere
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from Rag_sistem import RAGSystem  # Importar la clase RAGSystem

# Cargar variables de entorno
load_dotenv()

# Configuración de API Keys y credenciales
CONFIG = {
    "COHERE_API_KEY" : os.getenv('COHERE_API_KEY'),
    "PINECONE_API_KEY" : os.getenv('PINECONE_API_KEY'),
    "PINECONE_ENVIRONMENT" : os.getenv('PINECONE_ENVIRONMENT'),
    "INDEX_NAME" : "desafiofinal"
}

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los headers
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    message: str

# Inicializar el sistema RAG
rag = RAGSystem()

# Endpoints
@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        response = rag.query(request.message)
        return {
            "message": response
        }
    except Exception as e:
        logger.error(f"Error en la consulta RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en la consulta RAG: {str(e)}")    
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)