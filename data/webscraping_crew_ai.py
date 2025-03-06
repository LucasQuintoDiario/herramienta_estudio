import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from typing import Callable

# Cargar variables de entorno
load_dotenv()

# Configurar la herramienta de búsqueda
search_tool = SerperDevTool()

# Crear un agente para realizar búsquedas
researcher = Agent(
    role='Investigador Web',
    goal='Buscar y extraer información relevante de la web',
    backstory='Eres un experto en investigación web y análisis de datos',
    tools=[search_tool],
    verbose=True
)

# Crear una tarea de investigación
research_task = Task(
    description='Busca información sobre The Bridge y sus programas de formación',
    agent=researcher
)

# Crear y ejecutar el crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

# Ejecutar la investigación
result = crew.kickoff()
print(result) 