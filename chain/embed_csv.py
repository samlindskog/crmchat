import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.schema import Document
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSVEmbedder:
    def __init__(self, csv_dir: str = None, persist_dir: str = None):
        """
        Initialize the CSV embedder.
        
        Args:
            csv_dir: Directory containing CSV files (defaults to AIRTABLE_CSV_DIR env var or 'csv_output')
            persist_dir: Directory to persist vector store (defaults to AIRTABLE_VECTORSTORE_DIR env var or 'vectorstore')
        """
        # Load directories from environment variables with defaults
        self.csv_dir = Path(csv_dir or os.getenv('AIRTABLE_CSV_DIR', 'csv_output'))
        self.persist_dir = Path(persist_dir or os.getenv('AIRTABLE_VECTORSTORE_DIR', 'vectorstore'))
        
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create persist directory if it doesn't exist
        self.persist_dir.mkdir(exist_ok=True)
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        logger.info(f"CSV directory set to: {self.csv_dir.absolute()}")
        logger.info(f"Vector store directory set to: {self.persist_dir.absolute()}")
    
    def load_csv_files(self) -> List[Document]:
        """Load and process all CSV files in the directory."""
        documents = []
        
        for csv_file in self.csv_dir.glob("*.csv"):
            logger.info(f"Processing {csv_file.name}")
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Create a DataFrameLoader
                loader = DataFrameLoader(df, page_content_column=df.columns[0])
                
                # Load documents
                docs = loader.load()
                
                # Add metadata about the source
                for doc in docs:
                    doc.metadata.update({
                        "source": csv_file.name,
                        "table": csv_file.stem,
                        "record_id": doc.metadata.get("airtable_record_id", "unknown")
                    })
                
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {csv_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {str(e)}")
                continue
        
        return documents
    
    def create_embeddings(self, documents: List[Document]) -> Chroma:
        """Create embeddings and store them in Chroma."""
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(texts)} chunks")
        
        # Create and persist vector store
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        
        # Persist the vector store
        vectorstore.persist()
        logger.info(f"Vector store persisted to {self.persist_dir}")
        
        return vectorstore
    
    def load_existing_vectorstore(self) -> Chroma:
        """Load an existing vector store from disk."""
        if not (self.persist_dir / "chroma.sqlite3").exists():
            raise FileNotFoundError("No existing vector store found")
        
        vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings
        )
        logger.info(f"Loaded existing vector store from {self.persist_dir}")
        return vectorstore
    
    def update_embeddings(self):
        """Update embeddings for all CSV files."""
        # Load documents
        documents = self.load_csv_files()
        if not documents:
            logger.warning("No documents found to process")
            return
        
        # Create embeddings
        vectorstore = self.create_embeddings(documents)
        logger.info("Embeddings updated successfully")

def main():
    try:
        embedder = CSVEmbedder()
        embedder.update_embeddings()
    except Exception as e:
        logger.error(f"Error updating embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    main() 