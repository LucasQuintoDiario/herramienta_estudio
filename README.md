# herramienta_estudio
# 📚 Plataforma de Aprendizaje y Asistencia para Bootcamps - The Bridge

Bienvenido al repositorio oficial del proyecto desarrollado para **The Bridge**, una plataforma integral diseñada para asistir tanto a potenciales estudiantes como a alumnos inscritos en los bootcamps de la escuela.

## 🚀 Descripción del Proyecto

Este proyecto consta de **dos aplicaciones principales**:

1. **Chatbot Informativo** al que hemos nombrado **Bridgy**, en honor a la escuela🎓
   - Proporciona información sobre los distintos bootcamps de The Bridge, incluyendo:
     - Modalidades de estudio
     - Precios
     - Localidades y campus disponibles
     - Contenidos de cada bootcamp
     - Información de contacto y proceso de inscripción
o
   - Utiliza técnicas avanzadas de RAG (Retrieval-Augmented Generation) con datos embebidos en **Pinecone** y consultas a fuentes oficiales de The Bridge.
    - ¿? Proporciona información sobre los bootcamps, precios, sedes, modalidades y otros detalles de interés.¿?

2. **Tutor Virtual: Una Plataforma de Aprendizaje para Estudiantes**
**Tutor Virtual: Una Plataforma para Estudiantes** 📊
   - Aplicación donde los estudiantes registrados pueden acceder a herramientas avanzadas de apoyo en su formación en **Data Science**:
     - **📜 Generador de Tests:** Crea tests automáticos sobre cualquier tema de DS.
     - **📝 Evaluador de Tests:** Corrige respuestas y proporciona feedback detallado.
     - **📚 Explicador de Conceptos:** Proporciona explicaciones detalladas sobre cualquier tema de DS.
     - **🎤 Simulador de Entrevistas:** Role-playing con un "headhunter" ficticio.
     - **🧐 Evaluador de Entrevistas:** Analiza el desempeño y da recomendaciones de mejora al alumno.
     - **📊 Evaluador de Rendimiento:** Analiza tendencias en los tests realizados por el usuario.
     - **📖 Creador de Flashcards:** Genera tarjetas de memorización sobre cualquier tema.
     - **📚 Recomendador de Materiales:** Sugiere libros y recursos en línea para aprender más.   
   
## 🛠️ Tecnologías Utilizadas

Este proyecto ha sido desarrollado con las siguientes herramientas y tecnologías:

- **Backend:**
  - Python (FastAPI, NLTK, OpenAI, CrewAI, Cohere)
  - OpenSearch & Pinecone (Vector DB para RAG)
  - Google Cloud & AWS (Infraestructura y almacenamiento)
  - MySQL (Gestión de usuarios y datos)
  - Docker (Contenedores para despliegue eficiente)

- **Frontend:**
  - React (Interfaz de usuario)
  - CSS, HTML y JavaScript (Diseño y funcionalidad de la plataforma)

o   
- **Lenguaje principal**: Python 🐍
- **Frameworks & Librerías**:
  - FastAPI (API backend)
  - NLTK (Procesamiento de Lenguaje Natural)
  - Cohere y OpenAI (Modelos de lenguaje)
  - CrewAI (Agentes inteligentes)
  - Pinecone y OpenSearch (Búsqueda y almacenamiento de embeddings)
  - MySQL (Base de datos para almacenamiento de usuarios y tests)
  - React, CSS, HTML y JavaScript (Frontend de la plataforma)
- **Infraestructura & Herramientas**:
  - Docker 🐳 (Contenedores para despliegue)
  - AWS y Google Cloud (Hosting y procesamiento)

## Estructura del Proyecto 📂
```
📦 herramienta_estudio
├── chatbot   
│   ├── RAG (Sistema de Recuperación Aumentada con Generación)   
│   │   ├── data (PDFs y documentos de la escuela embebidos)   
│   │   ├── rag_system.py (Motor de búsqueda e IA)   
│   │   ├── rag_system_clean.py (Optimización del pipeline de RAG)   
│   ├── backend   
│   │   ├── server.py (API del chatbot)   
│   ├── public (Archivos estáticos)   
│   ├── src (Frontend basado en React)    
├── backend (Tutor IA y gestión de usuarios)
│   ├── app
│   │   ├── database (Gestión de base de datos SQL)
│   │   ├── static (Recursos web)
│   │   ├── templates (Frontend básico para administración)
│   │   ├── main.py (Servidor FastAPI)
│   │   ├── utils.py (Funciones auxiliares)
│   ├── Dockerfile
│   ├── requirements.txt
└── README.md
```
## 📌 Instalación y Ejecución

### 1️⃣ Clonar el Repositorio
```bash
git clone https://github.com/LucasQuintoDiario/herramienta_estudio.git
cd herramienta_estudio
```

### 2️⃣ Configurar Variables de Entorno
Se requiere un archivo `.env` con las siguientes claves:
```plaintext
OPENAI_API_KEY=tu_api_key_openai
COHERE_API_KEY=tu_api_key_cohere
PINECONE_API_KEY=tu_api_key
PINECONE_ENVIRONMENT=tu_environment
MYSQL_HOST=tu_host
MYSQL_USER=tu_usuario
MYSQL_PASSWORD=tu_password
INDEX_NAME=desafiofinal
```

### 3️⃣ Construcción y Ejecución con Docker
```bash
docker-compose up --build
```

### 4️⃣ Acceder a la Plataforma
- **Chatbot Bridgy:** `http://localhost:8080/`??
- **Tutor Virtual:** `http://localhost:3000`??

## Uso 🚀

- **Chatbot:** Se accede a través del frontend y responde preguntas sobre la escuela.
- **Plataforma para estudiantes:** La escuela te registra, inicias sesion con tus credenciales y la plataforma te permite, entre otras cosas, realizar tests, practicar entrevistas, recibir feedback y mejorar el aprendizaje.

## 👥 Equipo de Desarrollo

Este proyecto ha sido desarrollado por:
- **Borja Barber**
- **Daniel Garrido**
- **Yanelis Gonzalez**
- **Lucas Herranz**
- **Daniel Masana**
- **Juan Zubiaga**

## 📬 Contacto
Para cualquier consulta, sugerencia o colaboración, puedes abrir un **issue** en este repositorio o ponerte en contacto con el equipo de desarrollo.

📍 Repositorio: [GitHub - Herramienta de Estudio](https://github.com/LucasQuintoDiario/herramienta_estudio)

¡Gracias por tu interés en este proyecto! 🚀
