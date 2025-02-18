# vector_store.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
import glob
import json

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db", batch_size: int = 100):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.collections = {
            "problem_analysis": "problem_analysis_knowledge",
            "config_recommendation": "config_recommendation_knowledge"
        }
        
        # Initialize text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
        )

    def _load_markdown_files(self, directory: str) -> List[Dict[str, str]]:
        """Load all markdown files from a directory."""
        documents = []
        
        try:
            # Get all .md files in the directory
            md_files = glob.glob(os.path.join(directory, "*.md"))
            
            for file_path in md_files:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    file_name = os.path.basename(file_path)
                    documents.append({
                        "content": content,
                        "source": file_name,
                        "type": os.path.basename(os.path.dirname(directory))
                    })
            
            return documents
        except Exception as e:
            print(f"Error loading markdown files from {directory}: {str(e)}")
            return []
        
    

    def initialize_knowledge_base(self, source_directory: str = "./source_knowledge") -> Dict:
        """Initialize the vector store with knowledge base documents from both collections."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            results = {}
            
            # Process each collection
            for collection_type, collection_name in self.collections.items():
                # Load documents from corresponding directory
                directory_path = os.path.join(source_directory, collection_type)
                documents = self._load_markdown_files(directory_path)
                
                if not documents:
                    results[collection_type] = {
                        "status": "error",
                        "error": f"No documents found in {directory_path}"
                    }
                    continue

                # Process documents
                processed_docs = []
                processed_metadatas = []
                
                for doc in documents:
                    chunks = self.text_splitter.split_text(doc["content"])
                    for i, chunk in enumerate(chunks):
                        processed_docs.append(chunk)
                        processed_metadatas.append({
                            "source": doc["source"],
                            "chunk": i,
                            "type": doc["type"]
                        })

                # Initialize Chroma with documents
                db = Chroma.from_texts(
                    texts=processed_docs,
                    metadatas=processed_metadatas,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=collection_name
                )
                db.persist()
                
                results[collection_type] = {
                    "status": "success",
                    "documents_processed": len(processed_docs),
                    "source_files": len(documents)
                }
            
            return {
                "status": "success",
                "collections": results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def get_retriever(self, collection_type: str):
        """Get a retriever instance for a specific collection."""
        if collection_type not in self.collections:
            raise ValueError(f"Invalid collection type: {collection_type}")
            
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collections[collection_type]
        )
        
        return db.as_retriever(
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "maximal_marginal_relevance": True
            }
        )

    def add_documents(self, documents: List[Dict[str, str]], collection_type: str) -> Dict:
        """Add new documents to a specific collection."""
        try:
            if collection_type not in self.collections:
                return {
                    "status": "error",
                    "error": f"Invalid collection type: {collection_type}"
                }

            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collections[collection_type]
            )

            processed_docs = []
            processed_metadatas = []

            for doc in documents:
                chunks = self.text_splitter.split_text(doc['content'])
                for i, chunk in enumerate(chunks):
                    processed_docs.append(chunk)
                    processed_metadatas.append({
                        "source": doc['source'],
                        "chunk": i,
                        "type": collection_type
                    })

            db.add_texts(
                texts=processed_docs,
                metadatas=processed_metadatas
            )
            db.persist()
            
            return {
                "status": "success",
                "documents_added": len(processed_docs)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def process(self, command: Dict) -> Dict:
        """Process commands for the vector store."""
        try:
            action = command.get("action")
            if action == "initialize":
                source_dir = command.get("source_directory", "./source_knowledge")
                return self.initialize_knowledge_base(source_dir)
            elif action == "add_documents":
                collection_type = command.get("collection_type")
                documents = command.get("documents", [])
                return self.add_documents(documents, collection_type)
            else:
                return {
                    "status": "error",
                    "error": f"Invalid action: {action}"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

if __name__ == "__main__":

    # load api key
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # Example usage
    manager = VectorStoreManager()
    
    # Initialize knowledge base
    init_command = {
        "action": "initialize",
        "source_directory": "./source_knowledge"
    }
    
    result = manager.process(init_command)
    print("\nInitialization Result:")
    print(json.dumps(result, indent=2))
    
    # # Example of adding new documents
    # new_docs_command = {
    #     "action": "add_documents",
    #     "collection_type": "problem_analysis",
    #     "documents": [
    #         {
    #             "content": "Example new problem analysis document",
    #             "source": "new_doc.md"
    #         }
    #     ]
    # }
    
    # result = manager.process(new_docs_command)
    # print("\nAdd Documents Result:")
    # print(json.dumps(result, indent=2))