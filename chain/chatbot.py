import os
from pathlib import Path
from typing import List, Dict
import logging
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AirtableChatbot:
    def __init__(self, persist_dir: str = None):
        """
        Initialize the chatbot.
        
        Args:
            persist_dir: Directory containing the vector store (defaults to AIRTABLE_VECTORSTORE_DIR env var or 'vectorstore')
        """
        # Load directory from environment variable with default
        self.persist_dir = Path(persist_dir or os.getenv('AIRTABLE_VECTORSTORE_DIR', 'vectorstore'))
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self._load_vectorstore()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the chain
        self.chain = self._create_chain()
        
        logger.info(f"Vector store directory set to: {self.persist_dir.absolute()}")
    
    def _load_vectorstore(self) -> Chroma:
        """Load the vector store from disk."""
        if not (self.persist_dir / "chroma.sqlite3").exists():
            raise FileNotFoundError(
                "No vector store found. Please run embed_csv.py first."
            )
        
        vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings
        )
        logger.info(f"Loaded vector store from {self.persist_dir}")
        return vectorstore
    
    def _create_chain(self) -> ConversationalRetrievalChain:
        """Create the conversational retrieval chain."""
        # Custom prompt template
        template = """You are a helpful AI assistant that has access to Airtable data. 
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Chat History: {chat_history}
        Human: {question}
        Assistant:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
        
        # Create the chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0),
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        return chain
    
    def chat(self, question: str) -> str:
        """
        Get a response from the chatbot.
        
        Args:
            question: The user's question
            
        Returns:
            The chatbot's response
        """
        try:
            response = self.chain({"question": question})
            return response["answer"]
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"

def main():
    try:
        # Initialize the chatbot
        chatbot = AirtableChatbot()
        
        print("Airtable Chatbot initialized. Type 'quit' to exit.")
        print("Ask questions about your Airtable data:")
        
        while True:
            # Get user input
            question = input("\nYou: ").strip()
            
            # Check for quit command
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Get and print response
            response = chatbot.chat(question)
            print(f"\nAssistant: {response}")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 