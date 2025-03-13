import pymysql
import os
from dotenv import load_dotenv
load_dotenv()

username = os.getenv("BBDD_USERNAME")
password = os.getenv("BBDD_PASSWORD")
host = os.getenv("BBDD_HOST")
port = os.getenv("BBDD_port")

db = pymysql.connect(host=host,
                     user=username,
                     password=password,
                     cursorclass=pymysql.cursors.DictCursor
)

cursor = db.cursor()

# Crear las bases de datos
create_db = '''CREATE DATABASE IF NOT EXISTS users_registrados'''
create_db_consultas = '''CREATE DATABASE IF NOT EXISTS consultas'''
cursor.execute(create_db)
cursor.execute(create_db_consultas)

# Verificar las bases de datos
print("BBDD creadas:")
cursor.execute('SHOW DATABASES')
bbdd = cursor.fetchall()
print(bbdd)

cursor.connection.commit()

# Usar la base de datos correcta
use_db = '''USE consultas'''
cursor.execute(use_db)

try:
    create_table = '''
    CREATE TABLE IF NOT EXISTS consultas (
        id INT AUTO_INCREMENT PRIMARY KEY,
        pregunta TEXT,
        respuesta TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    '''
    cursor.execute(create_table)

    print("Tabla consultas creada")

except pymysql.MySQLError as err:
    print(f"❌ Error de MySQL en consultas: {err}")

# Cambiar a la base de datos users_registrados
use_db = '''USE users_registrados'''
cursor.execute(use_db)

try:
    # Crear tabla Verticales
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Verticales (
        ID_Vertical INT AUTO_INCREMENT PRIMARY KEY,
        vertical VARCHAR(100) NOT NULL
    )
    """)

    # Crear tabla Users
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Users (
        ID_User INT AUTO_INCREMENT PRIMARY KEY,
        ID_Vertical INT,
        Nombre VARCHAR(100) NOT NULL,
        Apellidos VARCHAR(100) NOT NULL,
        Fecha_nacimiento DATE NOT NULL,
        nombre_usuario VARCHAR(255) NOT NULL,
        password VARCHAR(255) NOT NULL,
        FOREIGN KEY (ID_Vertical) REFERENCES Verticales(ID_Vertical) ON DELETE SET NULL
    )
    """)

    # Crear tabla Concepts
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Concepts (
        ID_Concepts INT AUTO_INCREMENT PRIMARY KEY,
        ID_User INT NOT NULL,
        Concepto VARCHAR(255) NOT NULL,
        Explicacion TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ID_User) REFERENCES Users(ID_User) ON DELETE CASCADE
    )
    """)

    # Crear tabla Test
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Test (
        ID_Test INT AUTO_INCREMENT PRIMARY KEY,
        ID_User INT NOT NULL,
        Preguntas TEXT NOT NULL,
        Respuestas TEXT NOT NULL,
        Feedback TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ID_User) REFERENCES Users(ID_User) ON DELETE CASCADE
    )
    """)

    # Crear tabla Recomendador
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Recomendador (
        ID_Recom INT AUTO_INCREMENT PRIMARY KEY,
        ID_User INT NOT NULL,
        Recomendaciones TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ID_User) REFERENCES Users(ID_User) ON DELETE CASCADE
    )
    """)

    # Crear tabla Flashcards
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Flashcards (
        ID_Fcard INT AUTO_INCREMENT PRIMARY KEY,
        ID_User INT NOT NULL,
        Contenido TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ID_User) REFERENCES Users(ID_User) ON DELETE CASCADE
    )
    """)

    print("✅ Tablas creadas correctamente.")

except pymysql.MySQLError as err:
    print(f"❌ Error de MySQL users_registrados: {err}")

# Guardar cambios y cerrar conexión
db.commit()
cursor.close()
db.close()
