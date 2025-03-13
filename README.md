# herramienta_estudio
# 📚 Plataforma de Aprendizaje y Asistencia para Bootcamps - The Bridge

Bienvenido al repositorio oficial del proyecto desarrollado para **The Bridge**, una plataforma integral diseñada para asistir tanto a potenciales estudiantes como a alumnos inscritos en los bootcamps de la escuela.

## 🚀 Descripción del Proyecto

Este proyecto consta de **dos aplicaciones principales**:

1. **Chatbot Informativo** al que hemos nombrado **Bridgy**, en honor a la escuela🎓
   - Proporciona información sobre los distintos bootcamps de The Bridge, incluyendo:
     - Modalidades de estudio
     - Localidades y campus disponibles
     - Contenidos de cada bootcamp
     - Información de contacto y proceso de inscripción


2. **Tutor Virtual: Una Plataforma de Aprendizaje para Estudiantes**
**Tutor Virtual: Una Plataforma para Estudiantes** 📊
   - Aplicación donde los estudiantes registrados pueden acceder a herramientas avanzadas de apoyo en su formación en **Data Science**:
     - **📜 Generador y Evaluador de Tests:** Crea tests automáticos sobre cualquier tema de DS y corrige las respuestas proporcionando feedback detallado.
     - **📖 Creador de Flashcards:** Genera tarjetas de memorización sobre cualquier tema.
     - **📚 Explicador de Conceptos:** Proporciona explicaciones detalladas sobre cualquier tema de DS.
     - **📊 Evaluador de Rendimiento:** Analiza tendencias en los tests realizados por el usuario.
     - **📚 Recomendador de Materiales:** Sugiere libros y recursos en línea para aprender más.   
     - **🎤 Simulador y Evaluador de Entrevistas:** Role-playing con un "headhunter" ficticio que analiza el desempeño y da recomendaciones de mejora al alumno.

## 🛠️ Tecnologías Utilizadas

Este proyecto ha sido desarrollado con las siguientes herramientas y tecnologías:

- **Backend:**
  - Python (FastAPI, NLTK, OpenAI, CrewAI, Cohere)
  - OpenSearch & Pinecone (Vector DB para RAG)
  - MySQL (Gestión de usuarios y datos)

- **Frontend:**
  - React (Interfaz de usuario)
  - CSS, HTML y JavaScript (Diseño y funcionalidad de la plataforma)

- **Infraestructura & Herramientas**:
  - Docker 🐳 (Contenedores para despliegue)
  - AWS y Google Cloud (Hosting y procesamiento)

## Estructura del Proyecto 📂

```bash
herramienta_estudio/       # Directorio raíz del proyecto
│──  📂 chatbot/               # Módulo relacionado con el chatbot de la herramienta
│    │── backend/               # Backend principal del sistema
│    │   │── Dockerfile         # Configuración de Docker para el backend
│    │   │── get-pip.py         # Script para instalar pip
│    │   │── Rag_sistem.py      # Implementación del sistema RAG (Retrieval-Augmented Generation)
│    │   │── requirements.txt   # Dependencias del backend
│    │   │── server.py          # Servidor principal del backend
│    ├── public/            # Archivos públicos del backend
│    ├── RAG/               # Implementación del sistema RAG
│    │   ├── data/          # Almacenamiento de datos procesados
│    │   ├── rag_system_pinecone.py  # Integración con Pinecone para almacenamiento vectorial
│    │── src/                   # Código fuente del frontend
│    │   ├── components/Contenido/  # Componentes relacionados con el contenido
│    │   │   ├── Contenido.js   # Componente principal de contenido
│    │   ├── config/            # Configuración del frontend
│──  📂 database/              # Módulo de base de datos
│    ├── bdd.ipynb          # Notebook para exploración de datos
│    ├── create_db.py       # Script para creación de la base de datos
│──  📂 tutor_virtual/         # Módulo del tutor virtual basado en IA
│    ├── static/            # Archivos estáticos (CSS, JS)
│    ├── templates/         # Plantillas HTML
│    ├── .dockerignore      # Archivos ignorados por Docker
│    ├── Dockerfile         # Configuración de Docker para el tutor virtual
│    ├── main.py            # Servidor principal del tutor virtual
│    ├── requirements.txt   # Dependencias del tutor virtual
│    ├── utils.py           # Funciones auxiliares
│    ├── .gitignore         # Archivos ignorados por Git
│── README.md              # Documentación del proyecto
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
#### Tutor virtual
```bash
cd tutor_virtual
docker build -t tutor_virtual .
docker run -d -p 8080:8080 --env-file .env --name backend tutor_virtual

```

#### Chatbot
```bash
cd chatbot
docker build -t chatbot .
docker run -d -p 8080:8080 --env-file .env --name chatbot

```


### 4️⃣ Acceder a la Plataforma
-  `http://localhost:8080/`

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

¡Gracias por tu interés en este proyecto! 🚀
