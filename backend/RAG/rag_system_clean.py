import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import cohere
from typing import List, Tuple
import os
import logging
from dotenv import load_dotenv
import json
import requests
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import glob
import pandas as pd
import re
import unicodedata
from nltk.tokenize import sent_tokenize
import nltk
from nltk.corpus import stopwords
import string

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
OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT')
OPENSEARCH_USERNAME = os.getenv('OPENSEARCH_USERNAME')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD')
INDEX_NAME = "pdf-chunks"

if not all([COHERE_API_KEY, OPENSEARCH_ENDPOINT, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD]):
    raise ValueError("Missing required environment variables")

try:
    co = cohere.Client(COHERE_API_KEY)
    logger.info("Cohere client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Cohere client: {e}")
    raise

class TextCleaner:
    """Clase para limpiar y normalizar texto."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('spanish'))
        self.punctuation = string.punctuation + '¿¡'
        # Patrones comunes en PDFs
        self.pdf_patterns = {
            'broken_chars': r'f_([a-z]+)',  # Para f_inanciacion, etc.
            'line_breaks': r'-\s*\n',  # Guiones al final de línea
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
        """Normaliza el texto eliminando acentos y convirtiendo a minúsculas."""
        # Convertir a minúsculas
        text = text.lower()
        # Normalizar caracteres Unicode
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return text
    
    def remove_special_chars(self, text: str) -> str:
        """Elimina caracteres especiales y mantiene solo letras, números y puntuación básica."""
        # Mantener solo caracteres alfanuméricos y puntuación básica
        text = re.sub(r'[^\w\s.,!?;:¿¡]', ' ', text)
        # Eliminar caracteres especiales comunes en PDFs
        text = re.sub(self.pdf_patterns['special_chars'], ' ', text)
        # Eliminar caracteres de control
        text = re.sub(self.pdf_patterns['control_chars'], '', text)
        # Eliminar URLs y emails
        text = re.sub(self.pdf_patterns['url_pattern'], '', text)
        text = re.sub(self.pdf_patterns['email_pattern'], '', text)
        return text
    
    def clean_whitespace(self, text: str) -> str:
        """Limpia espacios en blanco múltiples y saltos de línea."""
        # Reemplazar múltiples espacios con uno solo
        text = re.sub(self.pdf_patterns['multiple_spaces'], ' ', text)
        # Eliminar espacios al inicio y final
        text = text.strip()
        return text
    
    def remove_page_numbers(self, text: str) -> str:
        """Elimina números de página y encabezados/pies de página comunes."""
        # Eliminar números de página
        text = re.sub(r'\b\d+\s*$', '', text)
        # Eliminar encabezados/pies de página comunes
        text = re.sub(self.pdf_patterns['headers_footers'], '', text, flags=re.MULTILINE)
        # Eliminar números de página en formato "Página X de Y"
        text = re.sub(self.pdf_patterns['page_numbers'], '', text, flags=re.IGNORECASE)
        return text
    
    def remove_section_headers(self, text: str) -> str:
        """Elimina encabezados de sección comunes."""
        return re.sub(self.pdf_patterns['section_headers'], '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    def fix_broken_words(self, text: str) -> str:
        """Corrige palabras rotas en el texto."""
        # Corregir palabras con f_ al inicio
        text = re.sub(self.pdf_patterns['broken_chars'], r'\1', text)
        # Corregir guiones al final de línea
        text = re.sub(self.pdf_patterns['line_breaks'], '', text)
        return text
    
    def remove_duplicates(self, text: str) -> str:
        """Elimina líneas duplicadas en el texto."""
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
        """Aplica todas las transformaciones de limpieza al texto."""
        # Normalizar texto
        text = self.normalize_text(text)
        # Corregir palabras rotas
        text = self.fix_broken_words(text)
        # Eliminar caracteres especiales
        text = self.remove_special_chars(text)
        # Eliminar números de página
        text = self.remove_page_numbers(text)
        # Eliminar encabezados de sección
        text = self.remove_section_headers(text)
        # Eliminar duplicados
        text = self.remove_duplicates(text)
        # Limpiar espacios en blanco
        text = self.clean_whitespace(text)
        return text

class RAGSystem:
    def __init__(self, chunk_size: int = 200):
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = None
        self.pdf_sources = []
        self.text_cleaner = TextCleaner()
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract and clean text from PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        try:
            print(f"Attempting to open PDF file: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                print("File opened successfully")
                reader = PyPDF2.PdfReader(file)
                print(f"PDF reader created. Number of pages: {len(reader.pages)}")
                
                # Extraer y limpiar texto de cada página
                cleaned_texts = []
                for i, page in enumerate(reader.pages):
                    print(f"Processing page {i+1}/{len(reader.pages)}")
                    text = page.extract_text()
                    # Limpiar el texto de la página
                    cleaned_text = self.text_cleaner.clean_text(text)
                    if cleaned_text.strip():  # Solo agregar si hay texto después de la limpieza
                        cleaned_texts.append(cleaned_text)
                
                # Unir todos los textos limpios
                final_text = " ".join(cleaned_texts)
                print(f"Successfully extracted and cleaned text from PDF: {pdf_path}")
                return final_text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            raise

    def extract_text_from_csv(self, csv_path: str) -> str:
        """Extract and clean text from CSV file."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
            
        try:
            print(f"Attempting to open CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            print("CSV file opened successfully")
            
            # Procesar el archivo de precios
            if "Precios.csv" in csv_path:
                # Crear un formato estructurado para los precios
                price_texts = []
                for _, row in df.iterrows():
                    bootcamp = row.get('Bootcamp', '')
                    precio = row.get('Precio', '')
                    modalidad = row.get('Modalidad', '')
                    if bootcamp and precio:
                        price_text = f"Bootcamp: {bootcamp}\nModalidad: {modalidad}\nPrecio: {precio}€"
                        price_texts.append(price_text)
                
                # Agregar un encabezado claro para la sección de precios
                final_text = "INFORMACIÓN DE PRECIOS DE LOS BOOTCAMPS:\n\n" + "\n\n".join(price_texts)
            else:
                # Para otros archivos CSV, mantener el procesamiento original
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
        """Split text into chunks of specified size, respecting sentence boundaries and logical sections."""
        try:
            # Dividir el texto en secciones lógicas
            sections = []
            
            # Si es un archivo de precios, mantener cada bootcamp como una sección
            if "Precios.csv" in file_name:
                sections = text.split("\n\n")
            else:
                # Para otros archivos, dividir por secciones lógicas
                sections = re.split(r'(?i)(?:capítulo|sección|tema|módulo)\s+\d+[.:]?\s*', text)
            
            sections = [s.strip() for s in sections if s.strip()]
            
            chunks = []
            current_chunk = []
            current_size = 0
            
            for section in sections:
                # Dividir la sección en oraciones
                sentences = re.split(r'[.!?]+', section)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                for sentence in sentences:
                    sentence_words = sentence.split()
                    sentence_size = len(sentence_words)
                    
                    # Si la oración actual hace que el chunk exceda el tamaño máximo,
                    # guardar el chunk actual y comenzar uno nuevo
                    if current_size + sentence_size > self.chunk_size and current_chunk:
                        chunk_text = " ".join(current_chunk)
                        # Solo agregar si el chunk no está vacío y no es muy corto
                        if len(chunk_text.split()) >= 20:  # Mínimo 20 palabras
                            chunks.append(chunk_text)
                            self.pdf_sources.append(file_name)
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            # Agregar el último chunk si existe y cumple con el tamaño mínimo
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) >= 20:  # Mínimo 20 palabras
                    chunks.append(chunk_text)
                    self.pdf_sources.append(file_name)
            
            # Eliminar chunks duplicados
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
        """Generate embeddings for text chunks."""
        try:
            embeddings = self.model.encode(chunks, convert_to_tensor=True)
            embeddings = embeddings.cpu().numpy()
            logger.info(f"Successfully generated embeddings for {len(chunks)} chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def create_opensearch_index(self):
        """Create OpenSearch index with vector search capabilities."""
        try:
            # Define the index mapping
            mapping = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param": {
                            "ef_search": 100
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "text_chunk": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 384  # Dimension for all-MiniLM-L6-v2
                        },
                        "source": {"type": "keyword"}  # Campo para almacenar el origen del chunk
                    }
                }
            }

            # Create the index
            response = requests.put(
                f"{OPENSEARCH_ENDPOINT}/{INDEX_NAME}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(mapping),
                auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD)
            )
            
            if response.status_code not in [200, 400]:  # 400 means index already exists
                raise Exception(f"Failed to create index: {response.text}")
            
            logger.info("Successfully created OpenSearch index")
        except Exception as e:
            logger.error(f"Error creating OpenSearch index: {e}")
            raise

    def index_chunks(self):
        """Index chunks and their embeddings in OpenSearch."""
        try:
            # Prepare bulk indexing request
            bulk_data = []
            for i, (chunk, embedding, source) in enumerate(zip(self.chunks, self.embeddings, self.pdf_sources)):
                bulk_data.append({
                    "index": {
                        "_index": INDEX_NAME,
                        "_id": str(i)
                    }
                })
                bulk_data.append({
                    "text_chunk": chunk,
                    "embedding": embedding.tolist(),
                    "source": source
                })

            # Perform bulk indexing
            response = requests.post(
                f"{OPENSEARCH_ENDPOINT}/_bulk",
                headers={"Content-Type": "application/json"},
                data="\n".join(json.dumps(item) for item in bulk_data) + "\n",
                auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD)
            )

            if response.status_code != 200:
                raise Exception(f"Failed to index chunks: {response.text}")

            logger.info(f"Successfully indexed {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            raise

    def search_similar_chunks(self, query: str, k: int = 5) -> List[Tuple[str, str]]:
        """Search for similar chunks using OpenSearch kNN search."""
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_tensor=False)[0].tolist()

            # Prepare kNN search query
            search_body = {
                "size": k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": k
                        }
                    }
                }
            }

            # Perform search
            response = requests.post(
                f"{OPENSEARCH_ENDPOINT}/{INDEX_NAME}/_search",
                headers={"Content-Type": "application/json"},
                data=json.dumps(search_body),
                auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD)
            )

            if response.status_code != 200:
                raise Exception(f"Search failed: {response.text}")

            # Extract text chunks and sources from results
            hits = response.json()["hits"]["hits"]
            results = [(hit["_source"]["text_chunk"], hit["_source"]["source"]) for hit in hits]
            
            # Eliminar chunks duplicados
            unique_results = []
            seen_chunks = set()
            for chunk, source in results:
                if chunk not in seen_chunks:
                    seen_chunks.add(chunk)
                    unique_results.append((chunk, source))
            
            logger.info(f"Successfully found {len(unique_results)} unique similar chunks")
            return unique_results
        except Exception as e:
            logger.error(f"Error searching for similar chunks: {e}")
            raise

    def build_prompt(self, query: str, relevant_chunks: List[Tuple[str, str]]) -> str:
        """Build prompt for LLM using relevant chunks."""
        try:
            # Separar chunks por tipo de información
            price_chunks = []
            module_chunks = []
            other_chunks = []
            
            for chunk, source in relevant_chunks:
                if "Precios.csv" in source:
                    price_chunks.append((chunk, source))
                elif "módulo" in chunk.lower() or "tema" in chunk.lower() or "capítulo" in chunk.lower():
                    module_chunks.append((chunk, source))
                else:
                    other_chunks.append((chunk, source))
            
            # Paso 1: Resumir los chunks relevantes
            summary_prompt = """Eres un experto en resumir información de manera concisa y clara.
            Debes eliminar redundancias y mantener solo la información más relevante.
            
            Resume de manera concisa la siguiente información, eliminando detalles redundantes.
            IMPORTANTE: El resumen debe estar en español y ser conciso.
            
            Instrucciones para el resumen:
            1. Elimina información duplicada
            2. Mantén solo los puntos más relevantes
            3. Usa un lenguaje claro y directo
            4. No incluyas detalles técnicos innecesarios
            5. Si hay información de precios, inclúyela de manera clara y estructurada
            6. Si hay información de módulos, organízala de manera lógica\n\n"""
            
            # Agregar chunks en orden de prioridad
            if price_chunks:
                summary_prompt += "INFORMACIÓN DE PRECIOS:\n"
                for chunk, source in price_chunks:
                    summary_prompt += f"{chunk}\n\n"
            
            if module_chunks:
                summary_prompt += "INFORMACIÓN DE MÓDULOS:\n"
                for chunk, source in module_chunks:
                    summary_prompt += f"{chunk}\n\n"
            
            for chunk, source in other_chunks:
                summary_prompt += f"[Fuente: {source}]\n{chunk}\n\n"
            
            # Obtener el resumen
            summary_response = co.generate(
                prompt=summary_prompt,
                max_tokens=150,  # Aumentado para manejar más información
                temperature=0.2,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE',
                model='command'
            )
            summary = summary_response.generations[0].text.strip()
            
            # Paso 2: Generar la respuesta final
            final_prompt = f"""Eres un asistente experto que responde en español.
            Debes mantener un tono profesional y claro en todas tus respuestas.
            Nunca respondas en inglés.
            Sé conciso y directo, evitando divagar o agregar información no solicitada.

            Basándote en el siguiente resumen, responde la consulta de manera concisa y directa:

            Resumen:
            {summary}

            Consulta: {query}

            Instrucciones para la respuesta:
            1. RESPONDE EN ESPAÑOL
            2. Sé conciso y directo (máximo 3-4 frases)
            3. No repitas información
            4. Enfócate en los puntos más relevantes
            5. Si la información no es suficiente, indícalo
            6. No incluyas citas literales del texto
            7. Usa un tono profesional y claro
            8. No divagues ni agregues información no solicitada
            9. Si la consulta es sobre precios, incluye la información de manera clara y estructurada
            10. Si la consulta es sobre módulos, organiza la información de manera lógica"""
            
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
                temperature=0.2,  # Temperatura más baja para respuestas más deterministas
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE',
                model='command'
            )
            logger.info("Successfully generated response from Cohere")
            return response.generations[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating response with Cohere: {e}")
            raise

    def process_file(self, file_path: str):
        """Process a single file (PDF or CSV)."""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)
            
            # Extract text based on file type
            if file_extension == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_extension == '.csv':
                text = self.extract_text_from_csv(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return
            
            # Split into chunks
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
            
            # Create OpenSearch index
            self.create_opensearch_index()
            
            # Index chunks in OpenSearch
            self.index_chunks()
            
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise

    def query(self, question: str) -> str:
        """Process a query and return the response."""
        try:
            # Search for relevant chunks
            relevant_chunks = self.search_similar_chunks(question)
            
            # Build prompt
            prompt = self.build_prompt(question, relevant_chunks)
            
            # Get response from LLM
            response = self.get_llm_response(prompt)
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

def main():
    # Example usage
    data_directory = "data"
    
    try:
        # Initialize RAG system
        logger.info("Starting RAG system initialization...")
        rag = RAGSystem()
        rag.initialize(data_directory)
        
        # Example queries
        questions = [
            "¿Qué es el bootcamp de Data Science?",
            "¿Cuáles son los requisitos para el bootcamp de Ciberseguridad?",
            "¿Qué se aprende en el bootcamp de Desarrollo Web Full Stack?"
        ]
        
        for question in questions:
            logger.info(f"Processing question: {question}")
            response = rag.query(question)
            print(f"\nPregunta: {question}")
            print(f"Respuesta: {response}")
            print("-" * 80)
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        print(f"Error: {e}")
        print(f"Please make sure your files exist in: {data_directory}")
    except ValueError as e:
        logger.error(f"Value error: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 