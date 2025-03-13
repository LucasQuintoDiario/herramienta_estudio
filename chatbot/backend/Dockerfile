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
# Exponer el puerto que usa la aplicación
EXPOSE 8080
# Comando para ejecutar la aplicación
CMD ["python", "server.py"]