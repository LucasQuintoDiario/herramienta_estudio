from fastapi import FastAPI
import os
import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from crewai import Task, Crew, Agent
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


app = FastAPI(
    title="API de Agentes Educativos",
    description="API para interactuar con agentes de CrewAI en educación.",
    version="1.0.0"
)

test_creator = Agent(
    role="Test Creator",
    goal="Generar preguntas de prueba sobre temas de Data Science",
    backstory="Experto en evaluación educativa con enfoque en ciencia de datos.",
    verbose=True,
    model="gpt-3.5-turbo"
)

test_evaluator = Agent(
    role="Test Evaluator",
    goal="Evaluar respuestas de los estudiantes y dar retroalimentación",
    backstory="Profesor con experiencia en calificación de exámenes.",
    verbose=True,
    model="gpt-3.5-turbo"
)

flashcard_generator = Agent(
    role="Flashcard Generator",
    goal="Crear tarjetas de memoria para reforzar conceptos clave",
    backstory="Especialista en aprendizaje activo y memorización.",
    verbose=True,
    model="gpt-3.5-turbo"
)

concept_explainer = Agent(
    role="Concept Explainer",
    goal="Explicar conceptos complejos de Data Science de forma sencilla",
    backstory="Docente con habilidades para explicar temas difíciles de manera clara.",
    verbose=True,
    model="gpt-3.5-turbo"
)

performance_analyzer = Agent(
    role="Performance Analyzer",
    goal="Analizar tendencias en los errores de los estudiantes para mejorar su aprendizaje.",
    backstory="Especialista en análisis de datos educativos y detección de patrones de aprendizaje.",
    verbose=True,
    model="gpt-3.5-turbo"
)

tutor_personalized = Agent(
    role="Personalized Tutor",
    goal="Recomendar material adicional según el progreso del estudiante.",
    backstory="Mentor con experiencia en aprendizaje adaptativo y personalización de estudios.",
    verbose=True,
    model="gpt-3.5-turbo"
)

agents = {
    "test_creator": test_creator,
    "test_evaluator": test_evaluator,
    "flashcard_generator": flashcard_generator,
    "concept_explainer": concept_explainer,
    "performance_analyzer": performance_analyzer,
    "tutor_personalized": tutor_personalized
}



# Definición de modelos Pydantic
class TestQuestionRequest(BaseModel):
    topic: str
    num_questions: Optional[int] = 5

class AnswerEvaluationRequest(BaseModel):
    user_answers: Dict[str, str]
    correct_answers: Dict[str, str]

class FlashcardRequest(BaseModel):
    topic: str
    num_flashcards: Optional[int] = 10
    concepts: Optional[List[str]] = None

class ConceptExplanationRequest(BaseModel):
    concept: str

class PerformanceAnalysisRequest(BaseModel):
    user_answers: Dict[str, str]
    correct_answers: Dict[str, str]

class RecomendationsRequest(BaseModel):
    topic: str


@app.post("/generate-test-questions")
async def generate_test_questions(request: TestQuestionRequest):
    try:
        task = Task(
            description=f"Genera {request.num_questions} preguntas sobre {request.topic}",
            agent=agents["test_creator"],
            expected_output="Las preguntas deben ser de tipo short answer"
        )
        
        crew = Crew(agents=[agents["test_creator"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return result.tasks_output[0].raw if result.tasks_output else "No se pudieron generar preguntas."
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate-answers")
async def evaluate_answers(request: AnswerEvaluationRequest):
    try:
        task = Task(
            description="Evalúa las respuestas del estudiante y proporciona retroalimentación detallada.",
            agent=agents["test_evaluator"],
            expected_output="El feedback proporcionará una explicación del concepto, la respuesta correcta y una sugerencia para mejorar."
        )
        
        crew = Crew(agents=[agents["test_evaluator"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return result.tasks_output[0].raw if result.tasks_output else "No se pudo evaluar el examen."
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-flashcards")
async def create_flashcards(request: FlashcardRequest):
    try:
        task = Task(
            description=f"Crea {request.num_flashcards} flashcards sobre {request.topic}, incluyendo un concepto y una definición.",
            agent=agents["flashcard_generator"],
            expected_output="Tarjetas con un concepto y una definición."
        )
        
        crew = Crew(agents=[agents["flashcard_generator"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return result.tasks_output[0].raw if result.tasks_output else "No se pudieron generar flashcards."
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain-concept")
async def explain_concept(request: ConceptExplanationRequest):
    try:
        task = Task(
            description=f"Explica el concepto de {request.concept} de manera clara y sencilla.",
            agent=agents["concept_explainer"],
            expected_output="Explicación clara y sencilla del concepto."
        )
        
        crew = Crew(agents=[agents["concept_explainer"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return result.tasks_output[0].raw if result.tasks_output else "No se pudo generar la explicación."
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-performance")
async def analyze_performance(request: PerformanceAnalysisRequest):
    try:
        task = Task(
            description="Analiza el desempeño del estudiante y proporciona recomendaciones.",
            agent=agents["performance_analyzer"],
            expected_output="Recomendaciones de mejora basadas en errores comunes."
        )
        
        crew = Crew(agents=[agents["performance_analyzer"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return result.tasks_output[0].raw if result.tasks_output else "No se pudo analizar el desempeño."
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend-materials")
async def recommend_materials(request: RecomendationsRequest):
    try:
        task = Task(
            description=f"Recomienda material adicional sobre {request.topic}.",
            agent=agents["tutor_personalized"],
            expected_output="Lista de recursos recomendados."
        )
        
        crew = Crew(agents=[agents["tutor_personalized"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        return result.tasks_output[0].raw if result.tasks_output else "No se pudieron generar recomendaciones."
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)