from utils import *


app = FastAPI(
    title="API de Agentes Educativos",
    description="API para interactuar con agentes de CrewAI en educación.",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obtener la ruta base del proyecto (herramienta_estudio)
BASE_DIR = Path(__file__).resolve().parent.parent  # Sube un nivel desde backend

# Montar archivos estáticos
static_path = BASE_DIR / "app" / "static"  # Apunta a backend/app/statics
if not static_path.exists():
    raise RuntimeError(f"El directorio de archivos estáticos no existe: {static_path}")

app.mount("/static", StaticFiles(directory=static_path), name="static")

# Configurar templates
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")



# Ruta para servir la página principal
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
        # 1️⃣ Generar preguntas
        task_generation = Task(
            description=f"Genera {request.num_questions} preguntas sobre {request.topic}, asegurando que cada pregunta sea clara y este bien formulada.",
            agent=AGENTS["test_creator"],
            expected_output="Lista de preguntas de evaluación teórica."
        )
        
        # 2️⃣ Validar preguntas con el supervisor
        task_supervision = Task(
            description=f"Revisa estas preguntas y confirma que sean relevantes para Data Science y de calidad: {task_generation.expected_output}",
            agent=AGENTS["content_supervisor"],
            expected_output="Confirmación de calidad o rechazo de calidad, en caso de rechazar incluir la palabra rechazado."
        )
        
        # Ejecutar Crew con ambos agentes
        crew = Crew(agents=[AGENTS["test_creator"], AGENTS["content_supervisor"]], tasks=[task_generation, task_supervision], verbose=True)
        result = crew.kickoff()
        
        # 3️⃣ Obtener la salida
        questions = result.tasks_output[0].raw if result.tasks_output else "No se pudieron generar preguntas."
        supervision_feedback = result.tasks_output[1].raw if result.tasks_output else "No se pudo verificar la calidad."
        
        if "rechazado" in supervision_feedback.lower():
            raise HTTPException(status_code=400, detail="No se pueden generar preguntas sobre ese concepto.")
        
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
        
        task = Task(
            description=f"Evalúa las respuestas: {request.student_answers} para las preguntas: {questions}",
            agent=AGENTS["test_evaluator"],
            expected_output="""
                El informe debe incluir un resumen conciso de las fortalezas y debilidades que tengo, con una clara orientación sobre cómo mejorar. 
                Evita explicaciones largas y usa un lenguaje accesible y directo. El informe debe ser organizado de la siguiente forma:
                
                1. **Resumen General**: Un resumen corto de cómo fue mi desempeño general.
                2. **Áreas de Mejora**: Identificar las áreas específicas donde cometí errores o tengo dificultades.
                3. **Sugerencias de Mejora**: Propuestas claras y concretas sobre cómo puedo mejorar en esas áreas.
                4. **Refuerzos Positivos**: Asegúrate de incluir lo que hice bien para mantener mi motivación.
                
                Mantén el informe breve y enfocado, sin exceso de texto pero siendo lo suficientemente claro sobre qué necesito mejorar y cómo hacerlo.
                El informe debe estar redactado en segunda persona para  que sea  personal hacia mi.
            """
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
        task_generation = Task(
            description=f"""
        Genera {request.num_flashcards} flashcards sobre {request.topic}. 
        Cada flashcard debe seguir este formato exacto:

        **Flashcard X**  
        *Pregunta:* [Pregunta aquí]  
        *Respuesta:* [Respuesta aquí]

        ---

        Por ejemplo:

        **Flashcard 1**  
        *Pregunta:* ¿Qué es un bucle for en Python?  
        *Respuesta:* Un bucle for en Python permite iterar sobre una secuencia (como una lista o un rango) para ejecutar un bloque de código varias veces.

        ---

        **Flashcard 2**  
        *Pregunta:* ¿Cómo se usa un bucle for para iterar sobre una lista en Python?  
        *Respuesta:* Se puede usar de la siguiente manera: `for item in lista:`, donde `lista` es la colección de elementos a iterar.

        ---

        **Flashcard 3**  
        *Pregunta:* ¿Qué función cumple la función `range()` en un bucle for?  
        *Respuesta:* La función `range(n)` genera una secuencia de números del 0 al n-1, útil para iterar un número fijo de veces, por ejemplo: `for i in range(5):`.

        ---

        Asegúrate de que cada flashcard esté numerada correctamente (Flashcard 1, Flashcard 2, etc.) y que se mantenga el formato con los guiones.
            - La información debe ser 100% relevante para Data Science.
            - No uses definiciones genéricas, prioriza explicaciones prácticas y ejemplos.
            """,
            agent=AGENTS["flashcard_generator"],
            expected_output="Flashcards con formato específico y relevante para Data Science."
        )
        task_supervision = Task(
            description=f"Revisa estas flashcards y confirma que sean relevantes para Data Science y de calidad: {task_generation.expected_output}",
            agent=AGENTS["content_supervisor"],
            expected_output="Confirmación de calidad o echazo de calidad, en caso de rechazar incluir la palabra rechazado."
        )

        crew = Crew(agents=[AGENTS["flashcard_generator"], AGENTS["content_supervisor"]], tasks=[task_generation, task_supervision], verbose=True)
        result = crew.kickoff()
        flashcards_raw = result.tasks_output[0].raw if result.tasks_output else "No se pudieron generar flashcards."
        supervision_feedback = result.tasks_output[1].raw if result.tasks_output else "No se pudo verificar la calidad."
        
        if "rechazado" in supervision_feedback.lower():
            raise HTTPException(status_code=400, detail="Las flashcards generadas no cumplen con los estándares de calidad.")
        
        # Parsear el contenido de las flashcards en formato estructurado
        flashcards = parse_flashcards(flashcards_raw)

        # Guardar en la base de datos
        db = get_db_connection()
        try:
            with db.cursor() as cursor:
                query = "INSERT INTO Flashcards (ID_User, Contenido) VALUES (%s, %s)"
                cursor.execute(query, (user_id, json.dumps(flashcards)))  # Guardamos el JSON de las flashcards
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
        task_generation = Task(
            description=f"""
            Explica el concepto '{request.concept}' de forma clara y didáctica relacionado con Data Science.
            
            - Usa un lenguaje sencillo pero técnico.
            - Proporciona un ejemplo práctico.
            - Evita explicaciones demasiado teóricas, enfócate en la aplicación en Data Science.
            """,            
            agent=AGENTS["concept_explainer"],
            expected_output="Explicación clara con un ejemplo práctico."
        )

        task_supervision = Task(
            description=f"Revisa este contenido y confirma que sean relevantes para Data Science y de calidad: {task_generation.expected_output}",
            agent=AGENTS["content_supervisor"],
            expected_output="Confirmación de calidad o echazo de calidad, en caso de rechazar incluir la palabra rechazado."
        )
        crew = Crew(agents=[AGENTS["concept_explainer"], AGENTS["content_supervisor"]], tasks=[task_generation, task_supervision], verbose=True)
        result = crew.kickoff()
        explanation = result.tasks_output[0].raw if result.tasks_output else "No se pudo generar la explicación."
        supervision_feedback = result.tasks_output[1].raw if result.tasks_output else "No se pudo verificar la calidad."
        
        if "rechazado" in supervision_feedback.lower():
            raise HTTPException(status_code=400, detail="El concepto solicitado no cumplen con los estándares de calidad.")
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
        task_generation = Task(
            description=f"""
            Recomienda {request.num_materials} recursos de aprendizaje sobre '{request.topic}' en Data Science.
            
            - Incluye libros, cursos online y artículos.
            - Solo sugiere fuentes confiables como libros académicos, Coursera, edX, etc.
            - Especifica por qué cada recurso es útil y para qué nivel (básico, intermedio, avanzado).
            """,
            agent=AGENTS["tutor_personalized"],
            expected_output="Lista de recursos bien estructurada."
        )

        task_supervision = Task(
            description=f"Revisa estos materiales y confirma que sean relevantes para Data Science y de calidad: {task_generation.expected_output}",
            agent=AGENTS["content_supervisor"],
            expected_output="Confirmación de calidad o sugerencias de mejora."
        )
        
        # Ejecutar Crew con ambos agentes
        crew = Crew(agents=[AGENTS["tutor_personalized"], AGENTS["content_supervisor"]], tasks=[task_generation, task_supervision], verbose=True)
        result = crew.kickoff()
        materials = result.tasks_output[0].raw if result.tasks_output else "No se pudieron generar recomendaciones."
        supervision_feedback = result.tasks_output[1].raw if result.tasks_output else "No se pudo verificar la calidad."
        
        if "rechazado" in supervision_feedback.lower():
            raise HTTPException(status_code=400, detail="El material generado no cumplen con los estándares de calidad.")
        
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
                query_nombre = "SELECT nombre FROM Users WHERE ID_User = %s"
                cursor.execute(query_nombre, (user_id,))
                nombre = cursor.fetchone()
                if not results:
                    raise HTTPException(status_code=404, detail="No hay suficiente información para generar el informe")
                feedback_list = [result["Feedback"] for result in results]
        except pymysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Error al consultar evaluaciones: {str(e)}")
        finally:
            db.close()

        # Analizar el desempeño con el agente
        task = Task(
            description=f"Analiza el desempeño de {nombre} basado en: {feedback_list}",
            agent=AGENTS["performance_analyzer"],
            expected_output=f"Un informe personalizado para {nombre} escrito en segunda persona (usando 'tú'). Identifica tus patrones de error basados en los feedbacks proporcionados por el agente 'evaluate-answers', y ofrécete recomendaciones prácticas y únicas para que mejores tu desempeño. Asegúrate de que el tono sea motivacional y diferente al análisis del agente 'evaluate-answers'."
        )
        crew = Crew(agents=[AGENTS["performance_analyzer"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        recommendations = result.tasks_output[0].raw if result.tasks_output else "No se pudo analizar el desempeño."
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/technical-interview")
async def technical_interview(request: TechnicalInterviewRequest, user_id: int = Depends(get_current_user)):
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Obtener historial o crear uno nuevo
        conversation_history = redis_client.get(f"interview:{conversation_id}")
        conversation_history = json.loads(conversation_history) if conversation_history else []

        # Manejo especial para la primera interacción
        affirmative_pattern = re.compile(r"^(sí|si|claro|por supuesto|vale|listo|ok|de acuerdo|seguro)[!.]?$", re.IGNORECASE)

        # Si es la primera interacción y el mensaje es afirmativo
        if len(conversation_history) == 0 and affirmative_pattern.match(request.message):
            conversation_history.append({"role": "user", "content": request.message})

            # Crear la primera pregunta del entrevistador
            first_question = "¡Genial! Empecemos con la primera pregunta. ¿Puedes contarme un poco sobre ti?"

            assistant_message = {"role": "assistant", "content": first_question}
            conversation_history.append(assistant_message)

            # Guardar en Redis
            redis_client.setex(f"interview:{conversation_id}", timedelta(hours=24), json.dumps(conversation_history))

            return TechnicalInterviewResponse(
                messages=[assistant_message],
                conversation_id=conversation_id
            )

        # Agregar mensaje al historial si no es la primera interacción
        new_message = {"role": "user", "content": request.message}
        conversation_history.append(new_message)

        # Crear la tarea del entrevistador
        interview_task = Task(
            description=f"""Analiza el siguiente mensaje del candidato y responde apropiadamente de manera amigable y cercana:
            '{request.message}'

            Historia de la conversación:
            {conversation_history}
            Las respuestas deben ser cortas y que incluyan una nueva pregunta para que el candidato pueda responder.""",
            agent=AGENTS["entrevistador_agent"],
            expected_output="Respuesta del entrevistador"
        )

        # Si hay suficientes mensajes, generar feedback
        if len(conversation_history) >= 10:
            evaluation_task = Task(
                description=f"""Evalúa la conversación completa y genera un feedback detallado:
                {conversation_history}""",
                agent=AGENTS["evaluador_agent"],
                expected_output="Feedback del entrevistador"
            )

            crew = Crew(
                agents=[AGENTS["entrevistador_agent"], AGENTS["evaluador_agent"]],
                tasks=[interview_task, evaluation_task],
                process=Process.sequential
            )
            result = crew.kickoff()

            response = result.tasks_output[0].raw if result.tasks_output else "No se pudo obtener respuesta."
            feedback = result.tasks_output[1].raw if result.tasks_output else None

            assistant_message = {"role": "assistant", "content": "Muchas gracias  por participar en el proceso, la entrevista ha finalizado. Más abajo encontrarás el feedback!"}
            conversation_history.append(assistant_message)

            # Guardar en Redis
            redis_client.setex(f"interview:{conversation_id}", timedelta(hours=24), json.dumps(conversation_history))

            return TechnicalInterviewResponse(
                messages=[assistant_message],
                feedback=feedback,
                conversation_id=conversation_id
            )

        # Si no hay suficientes mensajes, solo responder
        crew = Crew(
            agents=[AGENTS["entrevistador_agent"]],
            tasks=[interview_task],
            process=Process.sequential
        )
        result = crew.kickoff()

        response = result.tasks_output[0].raw if result.tasks_output else "No se pudo obtener respuesta."

        assistant_message = {"role": "assistant", "content": response}
        conversation_history.append(assistant_message)

        # Guardar en Redis
        redis_client.setex(f"interview:{conversation_id}", timedelta(hours=24), json.dumps(conversation_history))

        return TechnicalInterviewResponse(
            messages=[assistant_message],
            conversation_id=conversation_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)