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

class RAGSystem:
    def __init__(self, chunk_size: int = 200):
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = None
        self.pdf_sources = []  # Lista para mantener registro de qué chunk viene de qué archivo
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {e}")
            raise
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        try:
            print(f"Attempting to open PDF file: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                print("File opened successfully")
                reader = PyPDF2.PdfReader(file)
                print(f"PDF reader created. Number of pages: {len(reader.pages)}")
                text = ""
                for i, page in enumerate(reader.pages):
                    print(f"Processing page {i+1}/{len(reader.pages)}")
                    text += page.extract_text()
            print(f"Successfully extracted text from PDF: {pdf_path}")
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            raise

    def extract_text_from_csv(self, csv_path: str) -> str:
        """Extract text from CSV file."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
            
        try:
            print(f"Attempting to open CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            print("CSV file opened successfully")
            
            # Convertir todas las columnas a texto y unirlas
            text = ""
            for column in df.columns:
                text += f"Columna {column}:\n"
                text += df[column].astype(str).str.cat(sep='\n')
                text += "\n\n"
            
            print(f"Successfully extracted text from CSV: {csv_path}")
            return text
        except Exception as e:
            print(f"Error extracting text from CSV: {str(e)}")
            raise
    
    def split_into_chunks(self, text: str, file_name: str) -> List[str]:
        """Split text into chunks of specified size."""
        try:
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.chunk_size):
                chunk = " ".join(words[i:i+self.chunk_size])
                chunks.append(chunk)
                self.pdf_sources.append(file_name)  # Guardar el origen del chunk
            logger.info(f"Successfully split text into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text into chunks: {e}")
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
    
    def search_similar_chunks(self, query: str, k: int = 3) -> List[Tuple[str, str]]:
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
            
            logger.info(f"Successfully found {len(results)} similar chunks")
            return results
        except Exception as e:
            logger.error(f"Error searching for similar chunks: {e}")
            raise
    
    def build_prompt(self, query: str, relevant_chunks: List[Tuple[str, str]]) -> str:
        """Build prompt for LLM using relevant chunks."""
        try:
            prompt = "Usa la siguiente información para responder la consulta:\n\n"
            for chunk, source in relevant_chunks:
                prompt += f"[Fuente: {source}]\n{chunk}\n\n"
            prompt += f"Consulta: {query}"
            logger.info("Successfully built prompt")
            return prompt
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            raise
    
    def get_llm_response(self, prompt: str) -> str:
        """Get response from LLM (Cohere)."""
        try:
            response = co.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.7,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE'
            )
            logger.info("Successfully generated response from Cohere")
            return response.generations[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating response with Cohere: {e}")
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