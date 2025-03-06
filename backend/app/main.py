from fastapi import FastAPI, HTTPException
import cohere
import uvicorn
from fastapi.responses import HTMLResponse
import os
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
import pymysql
import uuid
import datetime
from crewai import Agent, Task, Crew
from typing import List

# host = os.getenv("BBDD_HOST")
# username = os.getenv("BBDD_USERNAME")
# password= os.getenv("BBDD_PASSWORD")
# database= os.getenv("BBDD_NAME")
# api_key_cohere = os.getenv("API_COHERE")
# session_id = str(uuid.uuid4())

app = FastAPI()


class ExamResponse(BaseModel):
    answers: List[str]

# 📌 Endpoint 1: Generar Flashcards y Preguntas
@app.get("/generate_study_material")
async def generate_study_material():
    global flashcards, questions, correct_answers

    # 🏃 Ejecutar CrewAI para generar flashcards y preguntas
    result = study_crew.kickoff()

    # 🔹 Simulación de generación de contenido (debería venir de CrewAI)
    flashcards = [
        {"concept": "Regresión Lineal", "description": "Método para predecir valores continuos."},
        {"concept": "Overfitting", "description": "Modelo demasiado ajustado a los datos de entrenamiento."},
        {"concept": "Gradient Descent", "description": "Optimización iterativa para minimizar errores."},
    ]
    questions = [
        {"question": "¿Qué es la regresión lineal?", "correct_answer": "Método para predecir valores continuos."},
        {"question": "¿Qué es el overfitting?", "correct_answer": "Cuando un modelo se ajusta demasiado a los datos de entrenamiento."},
        {"question": "¿Para qué se usa Gradient Descent?", "correct_answer": "Para minimizar errores en modelos de Machine Learning."},
    ]

    # Almacenar respuestas correctas
    correct_answers = {q["question"]: q["correct_answer"] for q in questions}

    return {"flashcards": flashcards, "questions": questions}

# 📌 Nueva función para que el corrector evalúe respuestas
def review_answers(user_answers):
    global questions, correct_answers

    feedback_list = []

    for i, (q, user_answer) in enumerate(zip(questions, user_answers)):
        correct_answer = correct_answers.get(q["question"], "")

        # Definir la tarea específica para el corrector
        review_task = Task(
            description=(
                f"El estudiante respondió: '{user_answer}' a la pregunta '{q['question']}'. "
                f"La respuesta correcta es '{correct_answer}'. "
                "Proporciona feedback motivador explicando en qué acertó o qué puede mejorar."
            ),
            agent=reviewer
        )

        # Crear un equipo con solo el corrector y la tarea de feedback
        review_crew = Crew(agents=[reviewer], tasks=[review_task], verbose=True)
        feedback = review_crew.kickoff()

        feedback_list.append(f"📌 Pregunta {i+1}: {feedback}")

    return feedback_list

# 📌 Endpoint 2: Evaluación con el Corrector Motivador
@app.post("/submit_exam")
async def submit_exam(user_responses: ExamResponse):
    global questions

    if not questions:
        raise HTTPException(status_code=400, detail="Primero genera las preguntas con /generate_study_material")

    # 📌 Obtener feedback del corrector
    feedback = review_answers(user_responses.answers)

    return {"feedback": feedback}
