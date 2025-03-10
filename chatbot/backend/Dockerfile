FROM python:3.11-slim

WORKDIR /app

# Copiar los archivos de requisitos primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar las dependencias del sistema necesarias para sentence-transformers y torch
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Configurar las variables de entorno
ENV COHERE_API_KEY="8Ur9ffDfdPJqvxC4yTNx2GItkaHrLWcPImOfssz2"
ENV PINECONE_API_KEY="pcsk_2Q5SR6_RwRvbb2vr3Yi372tm17eCo9CVSrBbsMgCa7QVwoWhFiPDEvqeDwm8KEdggX8zAd"
ENV PINECONE_ENVIRONMENT="us-east-1"
ENV INDEX_NAME="desafiofinal"

# Exponer el puerto que usa la aplicación
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "server.py"] 