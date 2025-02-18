# vector_store.py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import os
import json

class VectorStoreManager:
    def __init__(self, persist_directory: str = "../chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.collection_name = "fabric_knowledge"
        
        # Initialize text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def initialize_knowledge_base(self) -> None:
        """Initialize the vector store with example knowledge base documents."""
        # Example knowledge base documents
        knowledge_base = {
            "configuration_best_practices": """
            Hyperledger Fabric Configuration Best Practices:
            
            1. Block Configuration:
            - Optimal block size depends on network capacity and transaction volume
            - For high throughput networks: 100-500 transactions per block
            - For networks with strict latency requirements: 10-50 transactions per block
            - Block timeout: 2s for high performance, 5s for balanced systems
            
            2. Consensus Configuration:
            - Raft ordering service recommended for most deployments
            - Kafka recommended for very high throughput requirements
            - Maximum batch timeout: Should not exceed 2s for optimal performance
            
            3. Endorsement Policies:
            - Simple policies (e.g., ANY) for development
            - N of M policies for production (e.g., 2 of 3)
            - Consider organizational requirements and trust model
            
            4. Channel Configuration:
            - Separate channels for different business domains
            - Avoid too many channels (increases system overhead)
            - Consider cross-channel queries impact
            
            5. Resource Allocation:
            - CPU: Minimum 4 cores per peer
            - Memory: 8GB minimum for peers
            - Disk: SSD recommended, minimum 100GB
            - Network: 1Gbps minimum for production
            """,
            
            "performance_troubleshooting": """
            Hyperledger Fabric Performance Troubleshooting Guide:
            
            1. Low Transaction Throughput:
            - Check block size configuration
            - Verify endorsement policy complexity
            - Monitor system resources (CPU, Memory, Disk I/O)
            - Analyze network latency between nodes
            
            2. High Latency Issues:
            - Review block timeout settings
            - Check network connectivity
            - Analyze endorsement policy path
            - Monitor chaincode execution time
            
            3. Timeout Problems:
            - Increase operation timeouts
            - Check system resource utilization
            - Verify network stability
            - Review chaincode complexity
            
            4. Resource Bottlenecks:
            - Monitor CPU utilization (should be <80%)
            - Check memory usage and garbage collection
            - Verify disk I/O performance
            - Analyze network bandwidth utilization
            
            5. Common Solutions:
            - Increase block size for higher throughput
            - Optimize endorsement policies
            - Scale hardware resources
            - Tune batch timeout settings
            """,
            
            "scaling_guidelines": """
            Hyperledger Fabric Scaling Guidelines:
            
            1. Horizontal Scaling:
            - Add more peers for read scalability
            - Increase orderer nodes for fault tolerance
            - Deploy multiple channels for parallel processing
            - Use private data collections for data partitioning
            
            2. Vertical Scaling:
            - Increase CPU cores for transaction processing
            - Add memory for better caching
            - Upgrade to faster storage (NVMe SSDs)
            - Improve network capacity
            
            3. Configuration Scaling:
            - Adjust block size based on network capacity
            - Tune batch timeout for optimal throughput
            - Modify endorsement policies for performance
            - Configure cache sizes appropriately
            
            4. Network Optimization:
            - Implement service discovery
            - Use gossip protocol effectively
            - Optimize anchor peer selection
            - Configure leader election parameters
            
            5. Monitoring Requirements:
            - Track peer resource utilization
            - Monitor orderer performance
            - Analyze transaction latency
            - Measure chaincode execution time
            """
        }

        # Create documents from knowledge base
        documents = []
        for topic, content in knowledge_base.items():
            chunks = self.text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": topic,
                        "chunk": i
                    }
                })

        # Initialize Chroma with documents
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        db = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        db.persist()
        print(f"Initialized vector store with {len(texts)} documents")

    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Add new documents to the vector store.
        
        Args:
            documents: List of dicts with 'content' and 'source' keys
        """
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

        processed_docs = []
        processed_metadatas = []

        for doc in documents:
            chunks = self.text_splitter.split_text(doc['content'])
            for i, chunk in enumerate(chunks):
                processed_docs.append(chunk)
                processed_metadatas.append({
                    "source": doc['source'],
                    "chunk": i
                })

        db.add_texts(
            texts=processed_docs,
            metadatas=processed_metadatas
        )
        db.persist()
        print(f"Added {len(processed_docs)} new document chunks to vector store")

    def get_retriever(self):
        """Get a retriever instance for the vector store."""
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        return db.as_retriever(
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "maximal_marginal_relevance": True,
                "filter": None
            }
        )