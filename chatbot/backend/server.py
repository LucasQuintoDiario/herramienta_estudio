from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from typing import List, Dict, Optional
import cohere
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
import pymysql
import json
from Rag_sistem import RAGSystem  # Importar la clase RAGSystem

# Cargar variables de entorno
load_dotenv()

# Configuración de API Keys y credenciales
CONFIG = {
    "COHERE_API_KEY" : os.getenv('COHERE_API_KEY'),
    "PINECONE_API_KEY" : os.getenv('PINECONE_API_KEY'),
    "PINECONE_ENVIRONMENT" : os.getenv('PINECONE_ENVIRONMENT'),
    "INDEX_NAME" : "desafiofinal",
    "BBDD_USERNAME": os.getenv("BBDD_USERNAME"),
    "BBDD_PASSWORD": os.getenv("BBDD_PASSWORD"),
    "BBDD_HOST": os.getenv("BBDD_HOST"),
    "BBDD_PORT": int(os.getenv("BBDD_PORT", 3306)),  # Puerto por defecto 3306
    "BBDD_NAME": 'consultas'
}

def get_db_connection():
    return pymysql.connect(
        host=CONFIG["BBDD_HOST"],
        user=CONFIG["BBDD_USERNAME"],
        password=CONFIG["BBDD_PASSWORD"],
        database=CONFIG["BBDD_NAME"],
        port=CONFIG["BBDD_PORT"],
        cursorclass=pymysql.cursors.DictCursor
    )

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
        db = get_db_connection()
        try:
            with db.cursor() as cursor:
                query = "INSERT INTO consultas (pregunta, respuesta) VALUES (%s, %s)"
                cursor.execute(query, (json.dumps(request.message), response))
                db.commit()
            return {"message": response}
        except pymysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Error en la consulta RAG: {str(e)}")
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error en la consulta RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en la consulta RAG: {str(e)}")    

port = int(os.getenv("PORT", 8080))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)











