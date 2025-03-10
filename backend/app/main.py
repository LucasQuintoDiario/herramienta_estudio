from utils import *

app = FastAPI(
    title="API de Agentes Educativos",
    description="API para interactuar con agentes de CrewAI en educación.",
    version="1.0.0"
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
        
        # Evaluar las respuestas
        task = Task(
            description=f"Evalúa las respuestas: {request.student_answers} para las preguntas: {questions}",
            agent=AGENTS["test_evaluator"],
            expected_output="Informe de evaluación, análisis de errores y retroalimentación de forma resumida para que no haya excesivo texto, pero sea claro sobre aspectos a mejorar y como."
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
    - Cada flashcard debe tener una pregunta y una respuesta corta y clara.
    - La información debe ser 100% relevante para Data Science.
    - No uses definiciones genéricas, prioriza explicaciones prácticas y ejemplos.
    """,
            agent=AGENTS["flashcard_generator"],
            expected_output="Flashcards con concepto y definición."
        )

        task_supervision = Task(
            description=f"Revisa estas flashcards y confirma que sean relevantes para Data Science y de calidad: {task_generation.expected_output}",
            agent=AGENTS["content_supervisor"],
            expected_output="Confirmación de calidad o echazo de calidad, en caso de rechazar incluir la palabra rechazado."
        )

        crew = Crew(agents=[AGENTS["flashcard_generator"], AGENTS["content_supervisor"]], tasks=[task_generation, task_supervision], verbose=True)
        result = crew.kickoff()
        flashcards = result.tasks_output[0].raw if result.tasks_output else "No se pudieron generar flashcards."
        supervision_feedback = result.tasks_output[1].raw if result.tasks_output else "No se pudo verificar la calidad."
        
        if "rechazado" in supervision_feedback.lower():
            raise HTTPException(status_code=400, detail="Las flashcards generadas no cumplen con los estándares de calidad.")
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
                if not results:
                    raise HTTPException(status_code=404, detail="No hay suficiente información para generar el informe")
                feedback_list = [result["Feedback"] for result in results]
        except pymysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Error al consultar evaluaciones: {str(e)}")
        finally:
            db.close()

        # Analizar el desempeño con el agente
        task = Task(
            description=f"Analiza el desempeño basado en: {feedback_list}",
            agent=AGENTS["performance_analyzer"],
            expected_output="Informe con análisis de patrones de error basado en  el conjunto de feedbacks  proporcionados por el agente llamado evaluate-answers y recomendaciones estratégicas para mejorar el desempeño del estudiante, la respuesta debe unica y ser diferente a la respuesta proporcionada por el agente llamado evaluate-answers."
        )
        crew = Crew(agents=[AGENTS["performance_analyzer"]], tasks=[task], verbose=True)
        result = crew.kickoff()
        recommendations = result.tasks_output[0].raw if result.tasks_output else "No se pudo analizar el desempeño."
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)