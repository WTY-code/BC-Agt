# vector_store.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Optional
from dotenv import load_dotenv
import asyncio
from tqdm import tqdm
import os
import glob
import json

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db", batch_size: int = 100, 
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.batch_size = batch_size
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
        
    def _batch_process_documents(self, processed_docs: List[str], processed_metadatas: List[Dict], collection_name: str):
        """process documents by batch"""
        try:
            # initiate progress bar
            total_batches = (len(processed_docs) + self.batch_size - 1) // self.batch_size
            pbar = tqdm(total=total_batches, desc=f"Processing {collection_name}")

            # create Chroma instance with persistence
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )

            # Process in batches
            for i in range(0, len(processed_docs), self.batch_size):
                batch_docs = processed_docs[i:i + self.batch_size]
                batch_metadatas = processed_metadatas[i:i + self.batch_size]
                
                # Create Document objects
                documents = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(batch_docs, batch_metadatas)
                ]
                
                # add batch data
                db.add_documents(documents)
                
                pbar.update(1)

            pbar.close()
            return db

        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            raise e

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

                # init Chroma in batch
                db = self._batch_process_documents(
                    processed_docs,
                    processed_metadatas,
                    collection_name
                )
                
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

            # Create Document objects
            langchain_docs = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(processed_docs, processed_metadatas)
            ]
            
            # Add documents
            db.add_documents(langchain_docs)
            
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