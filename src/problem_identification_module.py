# problem_identification_module.py
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from vector_store import VectorStoreManager
from rag_generation import RAGQueryGenerator
from system_metrics import SystemMetrics
from typing import List, Dict
import json
import os

class ProblemIdentificationModule:
    def __init__(self, llm: ChatOpenAI, vector_store_manager: VectorStoreManager):
        self.llm = llm
        self.retriever = vector_store_manager.get_retriever()
        self.query_generator = RAGQueryGenerator()
        
        problem_template = """
        Analyze the following Hyperledger Fabric system metrics and identify potential problems:

        Performance Metrics:
        - TPS: {tps}
        - Latency: {latency}ms
        - Block Size: {block_size}
        - Concurrent Requests: {concurrent_requests}
        - Timeouts: {timeouts}

        Current Configuration:
        {current_config}

        Retrieved Knowledge Base Information:
        {context}

        Based on the above information, provide a comprehensive analysis following these steps:
        1. Identify performance bottlenecks
        2. Analyze configuration issues
        3. Evaluate resource utilization
        4. Consider system architecture implications

        Format your response as JSON with the following structure:
        {
            "problems": [
                {
                    "category": "performance|configuration|resource|architecture",
                    "description": "Detailed problem description",
                    "severity": "high|medium|low",
                    "impact": "Impact on system performance",
                    "related_metrics": ["affected_metric1", "affected_metric2"]
                }
            ],
            "root_causes": [
                {
                    "problem_ref": "Index of related problem",
                    "description": "Detailed root cause explanation",
                    "confidence": "high|medium|low",
                    "evidence": "Evidence from metrics and configuration"
                }
            ]
        }
        """
        
        self.problem_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=problem_template,
                input_variables=["tps", "latency", "block_size", "concurrent_requests", 
                               "timeouts", "current_config", "context"]
            )
        )

    def analyze_problem(self, metrics: SystemMetrics) -> Dict:
        """Identify system problems and analyze root causes using enhanced RAG."""
        queries = self.query_generator.generate_queries(
            {
                "tps": metrics.tps,
                "latency": metrics.latency,
                "block_size": metrics.block_size,
                "concurrent_requests": metrics.concurrent_requests,
                "timeouts": metrics.timeouts,
                "current_config": metrics.current_config
            },
            ["block_size", "consensus", "endorsement"]
        )
        
        all_docs = []
        seen_docs = set()
        for query in queries:
            docs = self.retriever.get_relevant_documents(query)
            for doc in docs:
                doc_hash = hash(doc.page_content)
                if doc_hash not in seen_docs:
                    all_docs.append(doc)
                    seen_docs.add(doc_hash)

        context = "\n\n".join([
            f"Reference {i+1}:\n{doc.page_content}\n\nSource: {doc.metadata.get('source', 'Unknown')}"
            for i, doc in enumerate(all_docs[:5])
        ])

        result = self.problem_chain.run(
            tps=metrics.tps,
            latency=metrics.latency,
            block_size=metrics.block_size,
            concurrent_requests=metrics.concurrent_requests,
            timeouts=metrics.timeouts,
            current_config=json.dumps(metrics.current_config, indent=2),
            context=context
        )

        return json.loads(result)

    def process(self, metrics_data: Dict) -> Dict:
        """Process metrics and return problem analysis."""
        try:
            metrics = SystemMetrics(**metrics_data)
            analysis = self.analyze_problem(metrics)
            return {
                "status": "success",
                "analysis": analysis
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

if __name__ == "__main__":
    # Example usage
    vector_store_manager = VectorStoreManager(persist_directory="../chroma_db")
    if not os.path.exists("../chroma_db"):
        vector_store_manager.initialize_knowledge_base()
    
    llm = ChatOpenAI(temperature=0)
    module = ProblemIdentificationModule(llm, vector_store_manager)
    
    example_metrics = {
        "tps": 100.5,
        "latency": 250.0,
        "block_size": 10,
        "concurrent_requests": 1000,
        "timeouts": 5,
        "current_config": {
            "block_size": 10,
            "consensus_type": "kafka",
            "endorsement_policy": "AND('Org1.member', 'Org2.member')"
        }
    }
    
    result = module.process(example_metrics)
    print(json.dumps(result, indent=2))