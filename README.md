# herramienta_estudio
```bash
├── 📂 backend/  # Lógica del servidor y API
│   ├── 📂 app/
│   │   ├── 📂 models/  # Definición de modelos de datos (Pydantic, DB)
│   │   ├── 📂 services/  # Lógica de negocio (RAG, LLMs, agentes)
│   │   ├── 📂 routes/  # Endpoints de la API (FastAPI)
│   │   ├── 📂 database/  # Configuración de base de datos (PostgreSQL, FAISS, Pinecone)
│   │   ├── 📂 tests/  # Pruebas unitarias
│   │   ├── main.py  # Punto de entrada de la API
│   │   ├── config.py  # Configuración (variables de entorno, API Keys)
│   ├── requirements.txt  # Dependencias de backend
│   ├── Dockerfile  # Para contenedorización
│   ├── README.md  # Documentación del backend
│
├── 📂 frontend/  # Interfaz de usuario
│   ├── app.py  # Punto de entrada de Streamlit
│   ├── requirements.txt  # Dependencias de frontend
│   ├── README.md  # Documentación del frontend
│
├── 📂 multiagent-system/  # Agentes y flujo de trabajo
│   ├── 📂 agents/  # Definición de agentes en CrewAI/LangGraph
│   ├── 📂 workflows/  # Flujos de interacción entre agentes
│   ├── run.py  # Script de ejecución
│   ├── README.md  # Documentación de la arquitectura multiagente
│
├── 📂 data/  # Datos para el RAG
│   ├── 📂 raw/  # Datos sin procesar
│   ├── 📂 processed/  # Datos listos para indexar en la base de datos vectorial
│   ├── ingest_data.py  # Script para cargar datos en FAISS/Pinecone
│   ├── README.md  # Explicación de los datos usados
│
├── 📂 deployment/  # Infraestructura y despliegue
│   ├── 📂 scripts/  # Scripts de despliegue automatizado
│   ├── docker-compose.yml  # Configuración de contenedores
│   ├── README.md  # Guía de despliegue
│
├── README.md  # Documentación general del proyecto
```