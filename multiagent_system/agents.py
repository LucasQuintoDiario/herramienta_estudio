from crewai import Agent
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Inicialización de agentes
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