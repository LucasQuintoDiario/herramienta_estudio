from crewai import Agent, Task, Crew
from openai import OpenAI
import os

# Configurar la API de OpenAI
os.environ["OPENAI_API_KEY"] = ""

# Definir los agentes

# Agente que genera exámenes
test_creator = Agent(
    role="Test Creator",
    goal="Generar preguntas de prueba sobre temas de Data Science",
    backstory="Experto en evaluación educativa con enfoque en ciencia de datos.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Agente que evalúa respuestas
test_evaluator = Agent(
    role="Test Evaluator",
    goal="Evaluar respuestas de los estudiantes y dar retroalimentación",
    backstory="Profesor con experiencia en calificación de exámenes.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Agente que crea tarjetas de memoria
flashcard_generator = Agent(
    role="Flashcard Generator",
    goal="Crear tarjetas de memoria para reforzar conceptos clave",
    backstory="Especialista en aprendizaje activo y memorización.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Agente que explica conceptos
concept_explainer = Agent(
    role="Concept Explainer",
    goal="Explicar conceptos complejos de Data Science de forma sencilla",
    backstory="Docente con habilidades para explicar temas difíciles de manera clara.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Agente que analiza el desempeño del estudiante
performance_analyzer = Agent(
    role="Performance Analyzer",
    goal="Analizar tendencias en los errores de los estudiantes para mejorar su aprendizaje.",
    backstory="Especialista en análisis de datos educativos y detección de patrones de aprendizaje.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Agente que recomienda material personalizado
tutor_personalized = Agent(
    role="Personalized Tutor",
    goal="Recomendar material adicional según el progreso del estudiante.",
    backstory="Mentor con experiencia en aprendizaje adaptativo y personalización de estudios.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Definir las tareas
task_create_test = Task(
    description="Genera un conjunto de 5 preguntas sobre regresión logística.",
    agent=test_creator,
    expected_output="Lista de 5 preguntas sobre regresión logística."
)

task_evaluate_test = Task(
    description="Evalúa las respuestas del estudiante y proporciona retroalimentación detallada.",
    agent=test_evaluator,
    expected_output="Evaluación con puntuación y comentarios sobre cada respuesta."
)

task_generate_flashcards = Task(
    description="Crea 10 tarjetas de memoria sobre diferentes conceptos de Data Science.",
    agent=flashcard_generator,
    expected_output="10 tarjetas de memoria con preguntas y respuestas sobre diferentes conceptos de Data Science."
)

task_explain_concept = Task(
    description="Explica el concepto de overfitting con ejemplos claros.",
    agent=concept_explainer,
    expected_output="Explicación clara y concisa del concepto de overfitting con al menos un ejemplo."
)

task_analyze_performance = Task(
    description="Analiza tendencias en los errores de los estudiantes y sugiere áreas de mejora.",
    agent=performance_analyzer,
    expected_output="Informe sobre patrones de errores y sugerencias de mejora."
)

task_recommend_material = Task(
    description="Recomienda material de estudio adicional según el desempeño del estudiante.",
    agent=tutor_personalized,
    expected_output="Lista de recursos personalizados basados en las necesidades del estudiante."
)

# Crear el Crew
teacher_assistant_crew = Crew(
    agents=[test_creator, test_evaluator, flashcard_generator, concept_explainer, performance_analyzer, tutor_personalized],
    tasks=[task_create_test, task_evaluate_test, task_generate_flashcards, task_explain_concept, task_analyze_performance, task_recommend_material]
)

# Crear el Crew
teacher_assistant_crew = Crew(
    agents=[test_creator, test_evaluator, flashcard_generator, concept_explainer, performance_analyzer, tutor_personalized],
    tasks=[task_create_test, task_evaluate_test, task_generate_flashcards, task_explain_concept, task_analyze_performance, task_recommend_material]
)

# Ejecutar las tareas
teacher_assistant_crew.kickoff()
