from crewai import Agent, Task, Crew
from openai import OpenAI
import os

# Configurar la API de OpenAI
os.environ["OPENAI_API_KEY"] = ""

# Definir los agentes

# Agente que genera exámenes
test_creator = Agent(
    role="Test Creator",
    goal="Generar preguntas de prueba sobre temas de Data Science que sean claras, relevantes y desafiantes.",
    backstory="Experto en evaluación educativa con enfoque en ciencia de datos, especializado en la creación de preguntas que evalúan tanto la comprensión teórica como la aplicación práctica.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Agente que evalúa respuestas
test_evaluator = Agent(
    role="Test Evaluator",
    goal="Evaluar respuestas de los estudiantes de manera justa y proporcionar retroalimentación constructiva que ayude a mejorar su comprensión.",
    backstory="Profesor con amplia experiencia en calificación de exámenes y tutoría, conocido por su enfoque detallado y su capacidad para identificar áreas de mejora.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Agente que crea tarjetas de memoria
flashcard_generator = Agent(
    role="Flashcard Generator",
    goal="Crear tarjetas de memoria efectivas que refuercen conceptos clave de Data Science de manera concisa y fácil de recordar.",
    backstory="Especialista en aprendizaje activo y técnicas de memorización, con experiencia en la creación de materiales educativos que facilitan el aprendizaje a largo plazo.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Agente que explica conceptos
concept_explainer = Agent(
    role="Concept Explainer",
    goal="Explicar conceptos complejos de Data Science de manera clara y accesible, utilizando ejemplos prácticos y analogías cuando sea necesario.",
    backstory="Docente con habilidades excepcionales para simplificar temas difíciles, conocido por su capacidad para hacer que los conceptos abstractos sean comprensibles para todos.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Agente que analiza el desempeño del estudiante
performance_analyzer = Agent(
    role="Performance Analyzer",
    goal="Identificar patrones en los errores de los estudiantes y proporcionar recomendaciones específicas para mejorar su rendimiento.",
    backstory="Especialista en análisis de datos educativos, con experiencia en la identificación de tendencias y la creación de estrategias de mejora basadas en datos.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Agente que recomienda material personalizado
tutor_personalized = Agent(
    role="Personalized Tutor",
    goal="Recomendar recursos de estudio personalizados que se ajusten a las necesidades individuales de cada estudiante, basándose en su progreso y áreas de dificultad.",
    backstory="Mentor con experiencia en aprendizaje adaptativo, conocido por su capacidad para personalizar el contenido educativo y maximizar el potencial de cada estudiante.",
    verbose=True,
    model="gpt-3.5-turbo"
)

# Definir las tareas
task_create_test = Task(
    description="Genera un conjunto de 5 preguntas sobre diferentes temas relacionados con data science. Las preguntas deben cubrir tanto aspectos teóricos como aplicaciones prácticas, y deben ser lo suficientemente desafiantes para evaluar la comprensión profunda del tema.",
    agent=test_creator,
    expected_output="Lista de 5 preguntas bien redactadas sobre temas relazionados con data science, con un equilibrio entre teoría y práctica."
)

task_evaluate_test = Task(
    description="Evalúa las respuestas del estudiante a las preguntas sobre data science. Proporciona una puntuación detallada y retroalimentación constructiva para cada respuesta, destacando tanto los aciertos como las áreas de mejora.",
    agent=test_evaluator,
    expected_output="Evaluación detallada con puntuación y comentarios específicos para cada respuesta, incluyendo sugerencias para mejorar."
)

task_generate_flashcards = Task(
    description="Crea 10 tarjetas de memoria sobre diferentes conceptos de Data Science. Cada tarjeta debe contener una pregunta clara en un lado y una respuesta concisa en el otro, enfocándose en conceptos clave que los estudiantes deben recordar.",
    agent=flashcard_generator,
    expected_output="10 tarjetas de memoria bien estructuradas, con preguntas y respuestas claras y relevantes sobre conceptos clave de Data Science."
)

task_explain_concept = Task(
    description="Explica el concepto de overfitting en el contexto de Machine Learning. Utiliza ejemplos prácticos y analogías para hacer que el concepto sea fácil de entender para estudiantes con diferentes niveles de conocimiento.",
    agent=concept_explainer,
    expected_output="Explicación clara y accesible del concepto de overfitting, con al menos un ejemplo práctico y una analogía que facilite la comprensión."
)

task_analyze_performance = Task(
    description="Analiza las tendencias en los errores de los estudiantes en las respuestas del examen sobre regresión logística. Identifica los patrones comunes y sugiere áreas de mejora específicas para cada estudiante.",
    agent=performance_analyzer,
    expected_output="Informe detallado que identifica los patrones de errores más comunes y proporciona recomendaciones específicas para mejorar el rendimiento de los estudiantes."
)

task_recommend_material = Task(
    description="Recomienda material de estudio adicional para un estudiante que ha tenido dificultades con el concepto de overfitting. Los recursos deben ser personalizados y adaptados a su nivel de comprensión actual.",
    agent=tutor_personalized,
    expected_output="Lista de recursos personalizados, incluyendo artículos, videos y ejercicios prácticos, que ayudarán al estudiante a mejorar su comprensión del concepto de overfitting."
)

# Crear el Crew
teacher_assistant_crew = Crew(
    agents=[test_creator, test_evaluator, flashcard_generator, concept_explainer, performance_analyzer, tutor_personalized],
    tasks=[task_create_test, task_evaluate_test, task_generate_flashcards, task_explain_concept, task_analyze_performance, task_recommend_material]
)

# Ejecutar las tareas
teacher_assistant_crew.kickoff()