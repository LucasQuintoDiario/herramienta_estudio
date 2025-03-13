FROM node:16-alpine

WORKDIR /app

# Copiar archivos de configuración
COPY package*.json ./

# Instalar dependencias
RUN npm install

# Copiar el código fuente
COPY . .

# Construir la aplicación
RUN npm run build

# Instalar servidor ligero para servir la aplicación
RUN npm install -g serve

# Exponer el puerto
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD serve -s build -l ${PORT:-8080} 