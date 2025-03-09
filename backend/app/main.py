from fastapi import FastAPI, HTTPException, Depends, Request
import os
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Optional
from crewai import Task, Crew, Agent
from opensearchpy import OpenSearch
import cohere
import logging
import pymysql
import redis
import uuid
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


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
    "BBDD_NAME": os.getenv("BBDD_NAME",'users_registrados')  # Nombre de la BBDD
}

# Inicializar clientes
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
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

# Modelos Pydantic (actualizados)
class LoginRequest(BaseModel):
    nombre_usuario: str
    password: str

class TestQuestionRequest(BaseModel):
    topic: str
    num_questions: Optional[int] = 5

class AnswerEvaluationRequest(BaseModel):
    student_answers: List[str]

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

# Verificación de token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    username = redis_client.get(token)
    if not username:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    db = get_db_connection()
    try:
        with db.cursor() as cursor:
            cursor.execute("SELECT ID_User FROM Users WHERE nombre_usuario = %s", (username,))
            result = cursor.fetchone()
            if not result:
                raise HTTPException(status_code=401, detail="Usuario no encontrado")
            return result["ID_User"]
    except pymysql.MySQLError as e:
        raise HTTPException(status_code=500, detail=f"Error de base de datos: {str(e)}")
    finally:
        db.close()

# Endpoints
@app.post("/login")
async def login(request: LoginRequest):
    db = get_db_connection()
    try:
        with db.cursor() as cursor:
            query = "SELECT ID_User, nombre FROM Users WHERE nombre_usuario = %s AND password = %s"
            cursor.execute(query, (request.nombre_usuario, request.password))
            result = cursor.fetchone()
            if result:
                session_token = str(uuid.uuid4())
                redis_client.setex(session_token, 3600, request.nombre_usuario)
                return {"status": "ok", "message": f"Hola, {result['nombre']}", "session_token": session_token}
            raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    except pymysql.MySQLError as e:
        raise HTTPException(status_code=500, detail=f"Error de base de datos: {str(e)}")
    finally:
        db.close()


@app.post("/generate-test-questions")
async def generate_test_questions(request: TestQuestionRequest, req: Request, user_id: int = Depends(get_current_user)):
    try:
        task = Task(
            description=f"Genera {request.num_questions} preguntas sobre {request.topic}",
            agent=AGENTS["test_creator"],
            expected_output="Lista de preguntas de evaluación teórica."
        )
        crew = Crew(agents=[AGENTS["test_creator"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        questions = result.tasks_output[0].raw if result.tasks_output else "No se pudieron generar preguntas."
        
        # Convertir questions a lista si es necesario
        if isinstance(questions, str):
            questions_list = questions.split("\n")  # Ajusta según el formato real
        else:
            questions_list = questions
        
        # Obtener el token desde los encabezados del Request
        token = req.headers.get("Authorization").split("Bearer ")[1]
        
        # Guardar las preguntas en Redis
        redis_client.setex(f"questions:{token}", 3600, json.dumps(questions_list))
        
        # Guardar en la base de datos
        db = get_db_connection()
        try:
            with db.cursor() as cursor:
                query = "INSERT INTO Test (ID_User, Preguntas, Respuestas, Feedback) VALUES (%s, %s, %s, %s)"
                cursor.execute(query, (user_id, json.dumps(questions_list), json.dumps([]), "Pendiente de evaluación"))
                db.commit()
                test_id = cursor.lastrowid
            return {"test_id": test_id, "questions": questions_list}
        except pymysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Error al guardar preguntas: {str(e)}")
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/evaluate-answers")
async def evaluate_answers(request: AnswerEvaluationRequest, req: Request, user_id: int = Depends(get_current_user)):
    try:
        # Obtener el token desde los encabezados del Request
        token = req.headers.get("Authorization").split("Bearer ")[1]
        
        # Recuperar las preguntas desde Redis
        questions_json = redis_client.get(f"questions:{token}")
        if not questions_json:
            raise HTTPException(status_code=404, detail="Preguntas no encontradas o sesión expirada")
        questions = json.loads(questions_json)
        
        # Evaluar las respuestas
        task = Task(
            description=f"Evalúa las respuestas: {request.student_answers} para las preguntas: {questions}",
            agent=AGENTS["test_evaluator"],
            expected_output="Informe de evaluación detallado con puntuaciones, análisis de errores y retroalimentación específica para cada respuesta de forma resumida para que no haya excesivo texto."
        )
        crew = Crew(agents=[AGENTS["test_evaluator"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        feedback = result.tasks_output[0].raw if result.tasks_output else "No se pudo evaluar el examen."
        
        # Actualizar en la base de datos
        db = get_db_connection()
        try:
            with db.cursor() as cursor:
                query = """
                    UPDATE Test 
                    SET Respuestas = %s, Feedback = %s 
                    WHERE ID_User = %s AND Preguntas = %s AND Feedback = 'Pendiente de evaluación'
                """
                cursor.execute(query, (json.dumps(request.student_answers), feedback, user_id, json.dumps(questions)))
                db.commit()
                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Test no encontrado o ya evaluado")
            return {"feedback": feedback}
        except pymysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Error al guardar evaluación: {str(e)}")
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-flashcards")
async def create_flashcards(request: FlashcardRequest, user_id: int = Depends(get_current_user)):
    try:
        task = Task(
            description=f"Crea {request.num_flashcards} flashcards sobre {request.topic}",
            agent=AGENTS["flashcard_generator"],
            expected_output="Flashcards con concepto y definición."
        )
        crew = Crew(agents=[AGENTS["flashcard_generator"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        flashcards = result.tasks_output[0].raw if result.tasks_output else "No se pudieron generar flashcards."
        
        # Guardar en la base de datos
        db = get_db_connection()
        try:
            with db.cursor() as cursor:
                query = "INSERT INTO Flashcards (ID_User, Contenido) VALUES (%s, %s)"
                cursor.execute(query, (user_id, json.dumps(flashcards)))
                db.commit()
            return {"flashcards": flashcards}
        except pymysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Error al guardar flashcards: {str(e)}")
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain-concept")
async def explain_concept(request: ConceptExplanationRequest, user_id: int = Depends(get_current_user)):
    try:
        task = Task(
            description=f"Explica el concepto de {request.concept} de manera clara y sencilla.",
            agent=AGENTS["concept_explainer"],
            expected_output="Explicación clara del concepto con ejemplos prácticos y analogías intuitivas que faciliten su comprensión a distintos niveles.."
        )
        crew = Crew(agents=[AGENTS["concept_explainer"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        explanation = result.tasks_output[0].raw if result.tasks_output else "No se pudo generar la explicación."
        
        # Guardar en la base de datos
        db = get_db_connection()
        try:
            with db.cursor() as cursor:
                query = "INSERT INTO Concepts (ID_User, Concepto, Explicacion) VALUES (%s, %s, %s)"
                cursor.execute(query, (user_id, request.concept, explanation))
                db.commit()
            return {"explanation": explanation}
        except pymysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Error al guardar explicación: {str(e)}")
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-materials")
async def recommend_materials(request: RecommendationsRequest, user_id: int = Depends(get_current_user)):
    try:
        task = Task(
            description=f"Sugerir {request.num_materials} materiales sobre {request.topic}.",
            agent=AGENTS["tutor_personalized"],
            expected_output="Lista de recursos de estudio personalizados, con explicaciones y ejercicios diseñados para mejorar la comprensión de los conceptos necesarios."
        )
        crew = Crew(agents=[AGENTS["tutor_personalized"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        materials = result.tasks_output[0].raw if result.tasks_output else "No se pudieron generar recomendaciones."
        
        # Guardar en la base de datos
        db = get_db_connection()
        try:
            with db.cursor() as cursor:
                query = "INSERT INTO Recomendador (ID_User, Recomendaciones) VALUES (%s, %s)"
                cursor.execute(query, (user_id, json.dumps(materials)))
                db.commit()
            return {"recommended_materials": materials}
        except pymysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Error al guardar recomendaciones: {str(e)}")
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-performance")
async def analyze_performance(user_id: int = Depends(get_current_user)):
    try:
        # Obtener datos de la base de datos
        db = get_db_connection()
        try:
            with db.cursor() as cursor:
                query = "SELECT Feedback FROM Test WHERE ID_User = %s"
                cursor.execute(query, (user_id,))
                results = cursor.fetchall()
                if not results:
                    raise HTTPException(status_code=404, detail="No hay evaluaciones para este usuario")
                feedback_list = [result["Feedback"] for result in results]
        except pymysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Error al consultar evaluaciones: {str(e)}")
        finally:
            db.close()

        # Analizar el desempeño con el agente
        task = Task(
            description=f"Analiza el desempeño basado en: {feedback_list}",
            agent=AGENTS["performance_analyzer"],
            expected_output="Informe con análisis de patrones de error y recomendaciones estratégicas para mejorar el desempeño del estudiante, la respuesta debe unica y ser diferente a la respuesta proporcionada por el agente llamado evaluate-answers."
        )
        crew = Crew(agents=[AGENTS["performance_analyzer"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        recommendations = result.tasks_output[0].raw if result.tasks_output else "No se pudo analizar el desempeño."
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)