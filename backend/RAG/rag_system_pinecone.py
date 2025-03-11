import PyPDF2
import numpy as np
import cohere
from typing import List, Tuple
import os
import logging
from dotenv import load_dotenv
import json
import requests
from pinecone import Pinecone, ServerlessSpec
import glob
import pandas as pd
import re
import unicodedata
from nltk.tokenize import sent_tokenize
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt/PY3/spanish.pickle')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
INDEX_NAME = "desafiofinal"

if not all([COHERE_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError("Missing required environment variables")

try:
    co = cohere.Client(COHERE_API_KEY)
    logger.info("Cohere client initialized successfully")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Pinecone initialized successfully")
except Exception as e:
    logger.error(f"Error initializing clients: {e}")
    raise

def search_thebridge_web(query: str) -> str:
    """Realiza una búsqueda en Google restringida a sitios de The Bridge."""
    search_url = f"https://www.google.com/search?q=site:thebridge.tech {query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        return f"Puedes consultar información en el siguiente enlace: {search_url}"
    else:
        return "No pude obtener información de la web en este momento."

class TextCleaner:
    """Clase para limpiar y normalizar texto."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('spanish'))
        self.punctuation = string.punctuation + '¿¡'
        # Patrones comunes en PDFs
        self.pdf_patterns = {
            'broken_chars': r'f_([a-z]+)',
            'line_breaks': r'-\s*\n',
            'page_numbers': r'página\s+\d+\s+de\s+\d+',
            'headers_footers': r'^.*?\|.*?\|.*?$',
            'control_chars': r'[\x00-\x1F\x7F-\x9F]',
            'special_chars': r'[_\-\–\—\…]',
            'multiple_spaces': r'\s+',
            'section_headers': r'^(?:capítulo|sección|tema|módulo)\s+\d+[.:]?\s*',
            'email_pattern': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'url_pattern': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        }
    
    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return text
    
    def remove_special_chars(self, text: str) -> str:
        text = re.sub(r'[^\w\s.,!?;:¿¡]', ' ', text)
        text = re.sub(self.pdf_patterns['special_chars'], ' ', text)
        text = re.sub(self.pdf_patterns['control_chars'], '', text)
        text = re.sub(self.pdf_patterns['url_pattern'], '', text)
        text = re.sub(self.pdf_patterns['email_pattern'], '', text)
        return text
    
    def clean_whitespace(self, text: str) -> str:
        text = re.sub(self.pdf_patterns['multiple_spaces'], ' ', text)
        text = text.strip()
        return text
    
    def remove_page_numbers(self, text: str) -> str:
        text = re.sub(r'\b\d+\s*$', '', text)
        text = re.sub(self.pdf_patterns['headers_footers'], '', text, flags=re.MULTILINE)
        text = re.sub(self.pdf_patterns['page_numbers'], '', text, flags=re.IGNORECASE)
        return text
    
    def remove_section_headers(self, text: str) -> str:
        return re.sub(self.pdf_patterns['section_headers'], '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    def fix_broken_words(self, text: str) -> str:
        text = re.sub(self.pdf_patterns['broken_chars'], r'\1', text)
        text = re.sub(self.pdf_patterns['line_breaks'], '', text)
        return text
    
    def remove_duplicates(self, text: str) -> str:
        lines = text.split('\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)
    
    def clean_text(self, text: str) -> str:
        text = self.normalize_text(text)
        text = self.fix_broken_words(text)
        text = self.remove_special_chars(text)
        text = self.remove_page_numbers(text)
        text = self.remove_section_headers(text)
        text = self.remove_duplicates(text)
        text = self.clean_whitespace(text)
        return text

class RAGSystem:
    def __init__(self, chunk_size: int = 200):
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = None
        self.pdf_sources = []
        self.text_cleaner = TextCleaner()
        self.memory= {}
        
        try:
            # Inicializar Cohere para embeddings
            self.co = cohere.Client(COHERE_API_KEY)
            logger.info("Cohere client initialized successfully for embeddings")
            
            # Initialize Pinecone with new syntax
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Create Pinecone index if it doesn't exist
            if INDEX_NAME not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=INDEX_NAME,
                    dimension=768,  # Dimension for embed-multilingual-v2.0
                    metric='cosine'
                )
            self.index = self.pc.Index(INDEX_NAME)
            logger.info(f"Connected to Pinecone index: {INDEX_NAME}")
        except Exception as e:
            logger.error(f"Error in initialization: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        try:
            print(f"Attempting to open PDF file: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                print("File opened successfully")
                reader = PyPDF2.PdfReader(file)
                print(f"PDF reader created. Number of pages: {len(reader.pages)}")
                
                cleaned_texts = []
                for i, page in enumerate(reader.pages):
                    print(f"Processing page {i+1}/{len(reader.pages)}")
                    text = page.extract_text()
                    cleaned_text = self.text_cleaner.clean_text(text)
                    if cleaned_text.strip():
                        cleaned_texts.append(cleaned_text)
                
                final_text = " ".join(cleaned_texts)
                print(f"Successfully extracted and cleaned text from PDF: {pdf_path}")
                return final_text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            raise

    def extract_text_from_csv(self, csv_path: str) -> str:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
            
        try:
            print(f"Attempting to open CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            print("CSV file opened successfully")
            
            if "Precios.csv" in csv_path:
                price_texts = []
                for _, row in df.iterrows():
                    bootcamp = row.get('Bootcamp', '')
                    precio = row.get('Precio', '')
                    modalidad = row.get('Modalidad', '')
                    if bootcamp and precio:
                        price_text = f"Bootcamp: {bootcamp}\nModalidad: {modalidad}\nPrecio: {precio}€"
                        price_texts.append(price_text)
                
                final_text = "INFORMACIÓN DE PRECIOS DE LOS BOOTCAMPS:\n\n" + "\n\n".join(price_texts)
            else:
                cleaned_texts = []
                for column in df.columns:
                    column_text = df[column].astype(str).str.cat(sep='\n')
                    cleaned_text = self.text_cleaner.clean_text(column_text)
                    if cleaned_text.strip():
                        cleaned_texts.append(f"Columna {column}:\n{cleaned_text}")
                
                final_text = "\n\n".join(cleaned_texts)
            
            print(f"Successfully extracted and cleaned text from CSV: {csv_path}")
            return final_text
        except Exception as e:
            print(f"Error extracting text from CSV: {str(e)}")
            raise

    def split_into_chunks(self, text: str, file_name: str) -> List[str]:
        try:
            sections = []
            
            if "Precios.csv" in file_name:
                sections = text.split("\n\n")
            else:
                sections = re.split(r'(?i)(?:capítulo|sección|tema|módulo)\s+\d+[.:]?\s*', text)
            
            sections = [s.strip() for s in sections if s.strip()]
            
            chunks = []
            current_chunk = []
            current_size = 0
            
            for section in sections:
                sentences = re.split(r'[.!?]+', section)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                for sentence in sentences:
                    sentence_words = sentence.split()
                    sentence_size = len(sentence_words)
                    
                    if current_size + sentence_size > self.chunk_size and current_chunk:
                        chunk_text = " ".join(current_chunk)
                        if len(chunk_text.split()) >= 20:
                            chunks.append(chunk_text)
                            self.pdf_sources.append(file_name)
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) >= 20:
                    chunks.append(chunk_text)
                    self.pdf_sources.append(file_name)
            
            unique_chunks = []
            seen_chunks = set()
            for chunk in chunks:
                if chunk not in seen_chunks:
                    seen_chunks.add(chunk)
                    unique_chunks.append(chunk)
            
            logger.info(f"Successfully split text into {len(unique_chunks)} unique chunks")
            return unique_chunks
        except Exception as e:
            logger.error(f"Error splitting text into chunks: {e}")
            raise

    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings using Cohere's embed-multilingual-v2.0 model."""
        try:
            # Generate embeddings in batches of 96 (Cohere's limit)
            batch_size = 96
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                response = self.co.embed(
                    texts=batch,
                    model='embed-multilingual-v2.0',
                    input_type='search_document'
                )
                all_embeddings.extend(response.embeddings)
            
            embeddings = np.array(all_embeddings)
            logger.info(f"Successfully generated embeddings for {len(chunks)} chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def index_chunks(self):
        """Index chunks and their embeddings in Pinecone."""
        try:
            # Prepare vectors for indexing
            vectors = []
            for i, (chunk, embedding, source) in enumerate(zip(self.chunks, self.embeddings, self.pdf_sources)):
                vector = {
                    'id': str(i),
                    'values': embedding.tolist(),
                    'metadata': {
                        'text': chunk,
                        'source': source
                    }
                }
                vectors.append(vector)
            
            # Index in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Successfully indexed {len(vectors)} vectors in Pinecone")
        except Exception as e:
            logger.error(f"Error indexing chunks in Pinecone: {e}")
            raise

    def search_similar_chunks(self, query: str, k: int = 5) -> List[Tuple[str, str]]:
        try:
            query_embedding = self.co.embed(
                texts=[query],
                model='embed-multilingual-v2.0',
                input_type='search_query'
            ).embeddings[0]

            results = self.index.query(
                vector=query_embedding,
                top_k=k * 2,
                include_metadata=True
            )

            chunks = [(match['metadata']['text'], match['metadata']['source']) 
                    for match in results['matches']]

            texts = [chunk[0] for chunk in chunks]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts + [query])
            cosine_similarities = (tfidf_matrix * tfidf_matrix.T).toarray()[-1, :-1]
            sorted_indices = cosine_similarities.argsort()[-k:][::-1]
            unique_results = [chunks[i] for i in sorted_indices]

            logger.info(f"Encontrados {len(unique_results)} fragmentos relevantes.")
            return unique_results
        except Exception as e:
            logger.error(f"Error en la búsqueda híbrida: {e}")
            raise

    def process_file(self, file_path: str):
        """Process a single file (PDF or CSV)."""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)
            
            if file_extension == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_extension == '.csv':
                text = self.extract_text_from_csv(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return
            
            chunks = self.split_into_chunks(text, file_name)
            self.chunks.extend(chunks)
            
            logger.info(f"Successfully processed file: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    def initialize(self, data_directory: str):
        """Initialize the RAG system with multiple files."""
        try:
            # Process all PDFs and CSVs in the directory
            pdf_files = glob.glob(os.path.join(data_directory, "*.pdf"))
            csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
            
            for file_path in pdf_files + csv_files:
                self.process_file(file_path)
            
            if not self.chunks:
                raise ValueError("No valid files were processed")
            
            # Generate embeddings for all chunks
            self.embeddings = self.generate_embeddings(self.chunks)
            
            # Index chunks in Pinecone
            self.index_chunks()
            
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise

    def query(self, question: str) -> str:
        try:
            relevant_chunks = self.search_similar_chunks(question)
            
            if not relevant_chunks:
                web_info = search_thebridge_web(question)
                relevant_chunks.append((web_info, "Fuente externa"))
            
            seen_chunks = set()
            filtered_chunks = []
            for chunk, source in relevant_chunks:
                if chunk not in seen_chunks:
                    seen_chunks.add(chunk)
                    filtered_chunks.append((chunk, source))
            
            memory_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in list(self.memory.items())[-3:]])
            prompt = f"{memory_text}\n\n{self.build_prompt(question, filtered_chunks)}"
            
            response = self.get_llm_response(prompt)
            
            self.memory[question] = response
            return response
        except Exception as e:
            logger.error(f"Error en query con memoria y análisis de relevancia: {e}")
            raise

    def build_prompt(self, query: str, relevant_chunks: List[Tuple[str, str]]) -> str:
        """Build prompt for LLM using relevant chunks."""
        try:
            summary = "\n\n".join([f"Fuente: {source}\n{chunk}" for chunk, source in relevant_chunks])
            final_prompt = f"""Eres un asistente experto que responde en español.
            Debes mantener un tono profesional y claro en todas tus respuestas.
            Nunca respondas en inglés.
            Sé conciso y directo, evitando divagar o agregar información no solicitada.

            Basándote en la siguiente información, responde la consulta de manera clara y concisa:
            Resumen de la base de datos:
            {summary}

            Nota importante: "Carreer Readiness" No es un bootcamp. Es un servicio adicional que ayuda a 
            mejorar su empleabilidad mediante asesoramiento.

            Si la información en la base de datos no es suficiente, realiza una búsqueda 
            sólo en estos sitios web:
            - https://thebridge.tech
            - https://thebridge.tech/campus-madrid/
            - https://thebridge.tech/campus-online/
            - https://thebridge.tech/campus-bilbao/
            - https://thebridge.tech/campus-valencia/
            - https://thebridge.tech/quienes-somos/
            - https://thebridge.tech/bootcamps/

            Consulta: {query}

            Instrucciones para la respuesta:
            1. RESPONDE EN ESPAÑOL.
            2. Sé conciso y directo (máximo 3-4 frases).
            3. No repitas información.
            4. Enfócate en los puntos más relevantes.
            5. Si la información no es suficiente, indícalo.
            6. No incluyas citas literales del texto.
            7. Usa un tono profesional y claro.
            8. No divagues ni agregues información no solicitada.
            9. Si la consulta es sobre precios, incluye la información de manera clara y estructurada.
            10. Si la consulta es sobre módulos, organiza la información de manera lógica.
            """
            
            logger.info("Successfully built prompt")
            return final_prompt
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            raise

    def get_llm_response(self, prompt: str) -> str:
        """Get response from LLM (Cohere)."""
        try:
            response = co.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.2,
                k=8,
                stop_sequences=[],
                return_likelihoods='NONE',
                model='command-r-plus'
            )
            logger.info("Successfully generated response from Cohere")
            return response.generations[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating response with Cohere: {e}")
            raise

def main():
    data_directory = "data"
    
    try:
        # Initialize RAG system
        logger.info("Starting RAG system initialization...")
        rag = RAGSystem()
        rag.initialize(data_directory)
        
        # Example queries
        questions = [
            "¿Cuál es el precio del bootcamp de Data Science?",
            "¿Qué módulos incluye el bootcamp de Ciberseguridad?",
            "Hola",
            "¿Cuántos Bootcamps hay?",
            "¿Qué modalidades hay disponibles para el bootcamp Full Stack?",
            "¿qué es un león?",
            "¿me duele la espalda, qué hago?",
            "¿cuánto es 2 + 2?"
        ]
        
        for question in questions:
            logger.info(f"Processing question: {question}")
            response = rag.query(question)
            print(f"\nPregunta: {question}")
            print(f"Respuesta: {response}")
            print("-" * 80)
            
               
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
    