from fastapi import FastAPI, HTTPException
import os
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Optional
from crewai import Task, Crew, Agent
from opensearchpy import OpenSearch
import cohere
import logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


# Configuración de API Keys y credenciales
CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENSEARCH_HOST": os.getenv("OPENSEARCH_HOST"),
    "OPENSEARCH_USERNAME": os.getenv("OPENSEARCH_USERNAME"),
    "OPENSEARCH_PASSWORD": os.getenv("OPENSEARCH_PASSWORD"),
    "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
    "INDEX_NAME": os.getenv("INDEX_NAME")
}

# Inicializar clientes
cohere_client = cohere.Client(CONFIG["COHERE_API_KEY"])
opensearch_client = OpenSearch(
    hosts=[CONFIG["OPENSEARCH_HOST"]],
    http_auth=(CONFIG["OPENSEARCH_USERNAME"], CONFIG["OPENSEARCH_PASSWORD"])
)

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Agentes Educativos",
    description="API para interactuar con agentes de CrewAI en educación.",
    version="1.0.0"
)

# Definición de agentes
AGENTS = {
    "test_creator": Agent(
        role="Test Creator",
        goal="Generar preguntas de prueba sobre temas de Data Science",
        backstory="Experto en evaluación educativa.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),
    "test_evaluator": Agent(
        role="Test Evaluator",
        goal="Evaluar respuestas de los estudiantes y dar retroalimentación",
        backstory="Profesor con experiencia en calificación de exámenes.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),
    "flashcard_generator": Agent(
        role="Flashcard Generator",
        goal="Crear tarjetas de memoria para reforzar conceptos clave",
        backstory="Especialista en aprendizaje activo y memorización.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),
    "concept_explainer": Agent(
        role="Concept Explainer",
        goal="Explicar conceptos complejos de Data Science de forma sencilla",
        backstory="Docente con habilidades didácticas.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),
    "performance_analyzer": Agent(
        role="Performance Analyzer",
        goal="Analizar tendencias en los errores de los estudiantes.",
        backstory="Especialista en análisis de datos educativos.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),
    "tutor_personalized": Agent(
        role="Personalized Tutor",
        goal="Recomendar material adicional según el progreso del estudiante.",
        backstory="Mentor con experiencia en aprendizaje adaptativo.",
        verbose=True,
        model="gpt-3.5-turbo"
    )
}

# Modelos Pydantic
class QueryRequest(BaseModel):
    question: str

class TestQuestionRequest(BaseModel):
    topic: str
    num_questions: Optional[int] = 5

class FlashcardRequest(BaseModel):
    topic: str
    num_flashcards: Optional[int] = 10

class ConceptExplanationRequest(BaseModel):
    concept: str

# Función para obtener embeddings con Cohere
def get_question_embedding(question: str):
    """Obtiene el embedding de una pregunta usando SentenceTransformer."""
    return embedding_model.encode(question).tolist()

# Clase del sistema RAG
class RAGSystem:
    def __init__(self):
        logger.info("Inicializando RAGSystem...")
        self.client = opensearch_client
        self.index = CONFIG["INDEX_NAME"]

    def query(self, question: str):
        """Busca la mejor respuesta en OpenSearch y genera una respuesta con OpenAI."""
        logger.info(f"Procesando pregunta: {question}")

        # Obtener embedding
        question_embedding = get_question_embedding(question)

        # Realizar la búsqueda en OpenSearch
        query = {
            "size": 3,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": question_embedding,
                        "k": 3
                    }
                }
            }
        }

        response = self.client.search(index=self.index, body=query)
        hits = response.get("hits", {}).get("hits", [])

        if not hits:
            return "No encontré información relevante."

        # Extraer textos relevantes
        retrieved_texts = [hit["_source"]["text_chunk"] for hit in hits]

        # Generar respuesta con OpenAI
        prompt = f"Basado en la siguiente información:\n\n{retrieved_texts}\n\nPregunta: {question}\nRespuesta:"
        return f"Respuesta generada basada en la información recuperada:\n{retrieved_texts}"

rag = RAGSystem()

# Endpoints
@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        response = rag.query(request.question)
        return {"Pregunta": request.question, "Respuesta": response}
    except Exception as e:
        logger.error(f"Error en la consulta RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en la consulta RAG: {str(e)}")

@app.post("/generate-test-questions")
async def generate_test_questions(request: TestQuestionRequest):
    try:
        task = Task(
            description=f"Genera {request.num_questions} preguntas sobre {request.topic}",
            agent=AGENTS["test_creator"],
            expected_output="Lista de preguntas tipo short answer."
        )
        crew = Crew(agents=[AGENTS["test_creator"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return {"questions": result.tasks_output[0].raw} if result.tasks_output else {"message": "No se pudieron generar preguntas."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-flashcards")
async def create_flashcards(request: FlashcardRequest):
    try:
        task = Task(
            description=f"Crea {request.num_flashcards} flashcards sobre {request.topic}",
            agent=AGENTS["flashcard_generator"],
            expected_output="Flashcards con concepto y definición."
        )
        crew = Crew(agents=[AGENTS["flashcard_generator"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return {"flashcards": result.tasks_output[0].raw} if result.tasks_output else {"message": "No se pudieron generar flashcards."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain-concept")
async def explain_concept(request: ConceptExplanationRequest):
    try:
        task = Task(
            description=f"Explica el concepto de {request.concept} de manera clara y sencilla.",
            agent=AGENTS["concept_explainer"],
            expected_output="Explicación clara del concepto."
        )
        crew = Crew(agents=[AGENTS["concept_explainer"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return {"explanation": result.tasks_output[0].raw} if result.tasks_output else {"message": "No se pudo generar la explicación."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
