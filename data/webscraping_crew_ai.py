import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from typing import Callable

# Cargar variables de entorno
#load_dotenv()

#Poner la API key de OpenAI
os.environ['OPENAI_API_KEY'] = ''
# Configurar la API key de Serper
os.environ['SERPER_API_KEY'] = ''  # Reemplaza con tu API key de Serper

# Configurar la herramienta de búsqueda
search_tool = SerperDevTool(
    n_results=15
)

# Crear un agente para realizar búsquedas
researcher = Agent(
    role='Escraper especializado en extraer información estructurada de academias de formación.',
    goal=f'Obtener y estructurar información relevante sobre la empresa The Bridge, sus bootcamps y masters, campus, blog y quienes somos.',
    backstory=f"""Fuiste creado para explorar el vasto mundo de la educación tecnológica, recolectando y 
    organizando datos de bootcamps para facilitar la toma de decisiones de futuros estudiantes. 
    Tu precisión y capacidad de estructuración hacen que la información sea fácil de analizar y comparar. 
    Eres meticuloso, sigues buenas prácticas de scraping y evitas sobrecargar servidores innecesariamente.""",
    model='gpt-3.5-turbo',
    tools=[search_tool],
    verbose=True
)

# Crear una tarea de investigación
research_task = Task(
    description=f"""Acceder a toda la información que haya sobre The Bridge | Bootcamps y Masters en inovación digital y analizar su estructura HTML. Extraer información relevante:
Quiénes somos, qué hacemos, qué ofrecemos, número de contacto, etc.
Accede a la web principal, a la de los diferentes bootcamps y masters, a los campus, a la de los eventos y a la de los blogs.
Limpiar y estructurar los datos en un formato útil (JSON, CSV, etc.).
Asegurar que el scraping cumple con las normativas del sitio (robots.txt, headers adecuados).
Retornar la información de manera organizada para su análisis posterior.""",
    expected_output="Toda la información relacionada con The Bridge en un formato estructurado.",
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