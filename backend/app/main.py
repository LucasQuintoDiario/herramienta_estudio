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

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


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
        goal="Diseñar preguntas de evaluación desafiantes y relevantes sobre Data Science, abarcando teoría y aplicaciones prácticas.",
        backstory="Especialista en evaluación educativa con un profundo conocimiento en ciencia de datos. Crea exámenes estructurados para medir comprensión teórica y habilidades analíticas.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),
    "test_evaluator": Agent(
        role="Test Evaluator",
        goal="Calificar respuestas con precisión y proporcionar retroalimentación clara y útil para mejorar la comprensión del estudiante.",
        backstory="Profesor con experiencia en la evaluación de exámenes de Data Science. Su método de calificación identifica fortalezas y áreas de mejora.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),
    "flashcard_generator": Agent(
        role="Flashcard Generator",
        goal="Generar tarjetas de memoria efectivas que ayuden a reforzar conceptos clave de Data Science de forma clara y memorable.",
        backstory="Experto en técnicas de aprendizaje activo y retención de información, con experiencia en la creación de material didáctico interactivo.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),
    "concept_explainer": Agent(
        role="Concept Explainer",
        goal="Explicar conceptos de Data Science de manera clara y accesible, utilizando ejemplos prácticos y analogías intuitivas.",
        backstory="Docente apasionado por simplificar temas complejos, facilitando la comprensión a estudiantes con distintos niveles de experiencia.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),
    "performance_analyzer": Agent(
        role="Performance Analyzer",
        goal="Identificar patrones en los errores de los estudiantes y proporcionar estrategias de mejora basadas en datos.",
        backstory="Especialista en análisis de datos educativos, con experiencia en detectar tendencias de desempeño y optimizar estrategias de aprendizaje.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),
    "tutor_personalized": Agent(
        role="Personalized Tutor",
        goal="Sugerir materiales de estudio personalizados que refuercen los conocimientos del estudiante según sus necesidades específicas.",
        backstory="Mentor en aprendizaje adaptativo, capaz de seleccionar recursos óptimos para cada estudiante con base en su rendimiento académico.",
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


class RecommendationsRequest(BaseModel):
    topic: str  
    num_materials: Optional[int] = 5

class PerformanceAnalysisRequest(BaseModel):
    student_id: str
    exam_results: list 
    num_recommendations: Optional[int] = 3

class AnswerEvaluationRequest(BaseModel):
    student_answers: List[str]  # Respuestas del estudiante
    questions: List[str]  # Respuestas correctas para comparar


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

        # Generar respuesta con OpenAI (por ejemplo, utilizando Cohere, OpenAI o cualquier otro LLM)
        prompt = f"""
        He recuperado la siguiente información relevante para responder la pregunta del usuario:
        {retrieved_texts}
        Por favor, responde a la pregunta de forma clara y concisa, sin repetir el texto original.
        Pregunta: {question}
        Respuesta:
        """
        
        # Aquí sería donde usarías un modelo para generar la respuesta basada en el prompt.
        # Este ejemplo es con Cohere, pero puede ser OpenAI o cualquier otro modelo.

        # Suponiendo que uses Cohere:
        response = cohere_client.generate(
            model="command-r-plus",  # Reemplaza por el modelo que estés usando
            prompt=prompt,
            max_tokens=200,
            temperature=0.3
        )
        
        # Devuelves solo el texto generado, que es la respuesta
        return response.generations[0].text.strip() if response.generations else "No se pudo generar una respuesta."


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
            expected_output="Lista de preguntas de evaluación con una combinación equilibrada de teoría y práctica, diseñadas para medir comprensión y habilidades."
        )
        crew = Crew(agents=[AGENTS["test_creator"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return {"questions": result.tasks_output[0].raw} if result.tasks_output else {"message": "No se pudieron generar preguntas."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/evaluate-answers")
async def evaluate_answers(request: AnswerEvaluationRequest):
    try:
        task = Task(
            description=f"Evalúa las respuestas del estudiante: {request.student_answers} a las siguientes preguntas {request.questions} y proporciona retroalimentación detallada.",
            agent=AGENTS["test_evaluator"],
            expected_output="Informe de evaluación detallado con puntuaciones, análisis de errores y retroalimentación específica para cada respuesta."
        )
        crew = Crew(agents=[AGENTS["test_evaluator"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        
        # Retorna el feedback en formato de diccionario
        return {"feedback": result.tasks_output[0].raw} if result.tasks_output else {"message": "No se pudo evaluar el examen."}
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
            expected_output="Explicación clara del concepto con ejemplos prácticos y analogías intuitivas que faciliten su comprensión a distintos niveles.."
        )
        crew = Crew(agents=[AGENTS["concept_explainer"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return {"explanation": result.tasks_output[0].raw} if result.tasks_output else {"message": "No se pudo generar la explicación."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/analyze-performance")
async def analyze_performance(request: PerformanceAnalysisRequest):
    try:
        task = Task(
            description="Analiza el desempeño del estudiante y proporciona recomendaciones.",
            agent=AGENTS["performance_analyzer"],
            expected_output="Informe con análisis de patrones de error y recomendaciones estratégicas para mejorar el desempeño del estudiante."
        )
        crew = Crew(agents=[AGENTS["performance_analyzer"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return {"recommendations": result.tasks_output[0].raw} if result.tasks_output else {"message": "No se pudo analizar el desempeño."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-materials")
async def recommend_materials(request: RecommendationsRequest):
    try:
        task = Task(
            description=f"Sugerir materiales de estudio personalizados para un estudiante con dificultades en conceptos relacionados con {request.topic} Incluir artículos, videos y ejercicios prácticos adecuados a su nivel.",
            agent=AGENTS["tutor_personalized"],
            expected_output="Lista de recursos de estudio personalizados, con explicaciones y ejercicios diseñados para mejorar la comprensión de los conceptos necesarios."
        )
        crew = Crew(agents=[AGENTS["tutor_personalized"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return {"recommended_materials": result.tasks_output[0].raw} if result.tasks_output else {"message": "No se pudieron generar recomendaciones."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
