from fastapi import FastAPI, HTTPException, Depends, Request
import os
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Optional
from crewai import Task, Crew, Agent
import logging
import pymysql
import redis
import uuid
import json
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List
from datetime import timedelta
from crewai.process import Process
import re


# Cargar variables de entorno
load_dotenv()

# Configuración de API Keys y credenciales
CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENSEARCH_HOST": os.getenv("OPENSEARCH_HOST"),
    "OPENSEARCH_USERNAME": os.getenv("OPENSEARCH_USERNAME"),
    "OPENSEARCH_PASSWORD": os.getenv("OPENSEARCH_PASSWORD"),
    "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
    "INDEX_NAME": os.getenv("INDEX_NAME"),
    "BBDD_USERNAME": os.getenv("BBDD_USERNAME"),
    "BBDD_PASSWORD": os.getenv("BBDD_PASSWORD"),
    "BBDD_HOST": os.getenv("BBDD_HOST"),
    "BBDD_PORT": int(os.getenv("BBDD_PORT", 3306)),  # Puerto por defecto 3306
    "BBDD_NAME": os.getenv("BBDD_NAME",'users_registrados'),  # Nombre de la BBDD
}

# Inicializar clientes
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Agentes Educativos",
    description="API para interactuar con agentes de CrewAI en educación.",
    version="1.0.0"
)

# Seguridad para autenticación con token
security = HTTPBearer()


# Definición de agentes
AGENTS = {
    "test_creator": Agent(
        role="Test Creator",
        goal="Diseñar preguntas de evaluación desafiantes y relevantes, abarcando teoría.",
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
        goal="Explicar conceptos de manera clara y accesible, utilizando ejemplos prácticos y analogías intuitivas.",
        backstory="Docente apasionado por simplificar temas complejos, facilitando la comprensión a estudiantes con distintos niveles de experiencia.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),


    "performance_analyzer": Agent(
        role="Performance Analyzer",
        goal="Analizar tus patrones de errores y ofrecerte estrategias personalizadas para que mejores tu aprendizaje basándome en datos. Mi misión es evaluar cómo te está yendo y ayudarte a brillar en tus puntos fuertes mientras trabajamos juntos en tus áreas de mejora.",
        backstory="Soy un especialista en análisis de datos educativos, con experiencia en detectar cómo aprenden los estudiantes como tú y en crear planes para que alcances tu máximo potencial. Me apasiona entender tus tendencias de desempeño y darte herramientas prácticas para que mejores cada día.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),


    "tutor_personalized": Agent(
        role="Personalized Tutor",
        goal="Sugerir materiales de estudio personalizados que refuercen los conocimientos del estudiante según sus necesidades específicas.",
        backstory="Mentor en aprendizaje adaptativo, capaz de seleccionar recursos óptimos para cada estudiante con base en su rendimiento académico.",
        verbose=True,
        model="gpt-3.5-turbo"
    ),

    "content_supervisor": Agent(
    role="Content Supervisor",
    goal="Revisar y validar que el contenido generado esté relacionado con Data Science y tenga calidad.",
    backstory="Experto en Data Science y educación. Se asegura de que los materiales generados sean relevantes, precisos y útiles.",
    verbose=True,
    model="gpt-3.5-turbo"
    ),
   "entrevistador_agent": Agent(
    role='Entrevistador de Recursos Humanos Senior experto en evaluar soft skills',
    goal='Evaluar las soft skills del candidato',
    backstory="""Eres un entrevistador senior con amplia experiencia en Recursos Humanos.
    Tu objetivo es evaluar las soft skills del candidato a través de preguntas
    relevantes y casos prácticos.""",
    verbose=True,
    allow_delegation=False,
    llm_model="gpt-3.5-turbo",
    ),

    "evaluador_agent":Agent(
    role='Evaluador de Competencias',
    goal='Analizar las respuestas del candidato y proporcionar feedback detallado al candidato',
    backstory="""Eres un evaluador experto que analiza las respuestas de los candidatos
    para determinar su nivel de conocimiento, capacidad de resolución de problemas y
    áreas de mejora.""",
    verbose=True,
    allow_delegation=False,
    llm_model="gpt-3.5-turbo"
)
}



# Modelos Pydantic (actualizados)
class LoginRequest(BaseModel):
    nombre_usuario: str
    password: str

class TestQuestionRequest(BaseModel):
    topic: str
    num_questions: Optional[int] = 5

class AnswerEvaluationRequest(BaseModel):
    student_answers: str

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

# Modelos adicionales para la entrevista técnica
class TechnicalInterviewRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class TechnicalInterviewResponse(BaseModel):
    messages: List[dict]
    feedback: Optional[str] = None
    conversation_id: str



# Función para conectar a la base de datos
def get_db_connection():
    return pymysql.connect(
        host=CONFIG["BBDD_HOST"],
        user=CONFIG["BBDD_USERNAME"],
        password=CONFIG["BBDD_PASSWORD"],
        database=CONFIG["BBDD_NAME"],
        port=CONFIG["BBDD_PORT"],
        cursorclass=pymysql.cursors.DictCursor
    )


# Función para parsear las flashcards en formato estructurado
def parse_flashcards(raw_flashcards: str):
    flashcards = []
    # Asumiendo que las flashcards vienen como texto con el formato 'Flashcard X', vamos a dividirlas
    parts = raw_flashcards.split('---')  # Separar cada flashcard
    for part in parts:
        if part.strip():  # Ignorar los vacíos
            # Extraer la pregunta y respuesta usando expresiones regulares o búsqueda de texto
            question_start = part.find("*Pregunta:*") + len("*Pregunta:*")
            answer_start = part.find("*Respuesta:*") + len("*Respuesta:*")
            
            if question_start != -1 and answer_start != -1:
                question = part[question_start:part.find("*Respuesta:*")].strip()
                answer = part[answer_start:].strip()
                
                flashcards.append({
                    "pregunta": question,
                    "respuesta": answer
                })
    
    return flashcards