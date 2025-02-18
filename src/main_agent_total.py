# main_agent.py
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from vector_store import VectorStoreManager
from rag_generation import RAGQueryGenerator
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any
import json, os

@dataclass
class SystemMetrics:
    tps: float
    latency: float
    block_size: int
    concurrent_requests: int
    timeouts: int
    current_config: Dict[str, Any]

class SystemInfoAcquisitionModule:
    def __init__(self):
        self.metrics_schema = {
            "tps": float,
            "latency": float,
            "block_size": int,
            "concurrent_requests": int,
            "timeouts": int,
            "current_config": dict
        }

    def validate_metrics(self, raw_metrics: Dict) -> bool:
        """Validate if the input metrics match the required schema."""
        try:
            for key, type_class in self.metrics_schema.items():
                if key not in raw_metrics:
                    return False
                if not isinstance(raw_metrics[key], type_class):
                    return False
            return True
        except Exception:
            return False

    def extract_metrics(self, raw_metrics: Dict) -> SystemMetrics:
        """Extract and transform relevant metrics from raw data."""
        if not self.validate_metrics(raw_metrics):
            raise ValueError("Invalid metrics format")

        return SystemMetrics(
            tps=raw_metrics["tps"],
            latency=raw_metrics["latency"],
            block_size=raw_metrics["block_size"],
            concurrent_requests=raw_metrics["concurrent_requests"],
            timeouts=raw_metrics["timeouts"],
            current_config=raw_metrics["current_config"]
        )

class ProblemIdentificationModule:
    def __init__(self, llm: ChatOpenAI, vector_store_manager: VectorStoreManager):
        self.llm = llm
        self.retriever = vector_store_manager.get_retriever()
        self.query_generator = RAGQueryGenerator()
        
        # Initialize LLM chain for problem analysis
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
        # Generate multiple specialized queries
        queries = self.query_generator.generate_queries(
            {
                "tps": metrics.tps,
                "latency": metrics.latency,
                "block_size": metrics.block_size,
                "concurrent_requests": metrics.concurrent_requests,
                "timeouts": metrics.timeouts,
                "current_config": metrics.current_config
            },
            ["block_size", "consensus", "endorsement"]  # Example problem areas
        )
        
        # Retrieve and combine relevant documents from multiple queries
        all_docs = []
        seen_docs = set()
        for query in queries:
            docs = self.retriever.get_relevant_documents(query)
            for doc in docs:
                # Avoid duplicate documents using content hash
                doc_hash = hash(doc.page_content)
                if doc_hash not in seen_docs:
                    all_docs.append(doc)
                    seen_docs.add(doc_hash)

        # Combine retrieved contexts with importance weighting
        context = "\n\n".join([
            f"Reference {i+1}:\n{doc.page_content}\n\nSource: {doc.metadata.get('source', 'Unknown')}"
            for i, doc in enumerate(all_docs[:5])  # Limit to top 5 most relevant docs
        ])

        # Run LLM chain for problem analysis
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

class ConfigurationRecommendationModule:
    def __init__(self, llm: ChatOpenAI, vector_store_manager: VectorStoreManager):
        self.llm = llm
        self.retriever = vector_store_manager.get_retriever()
        self.query_generator = RAGQueryGenerator()

        config_template = """
        Based on the identified problems and current configuration, recommend parameter adjustments:

        Problem Analysis:
        {problem_analysis}

        Current Configuration:
        {current_config}

        Retrieved Knowledge Base Information:
        {context}

        Consider the following aspects when making recommendations:
        1. Impact on system performance
        2. Implementation complexity
        3. Resource requirements
        4. Potential risks and mitigation strategies
        5. Dependencies between parameters

        Generate configuration recommendations as JSON with the following structure:
        {
            "recommendations": [
                {
                    "parameter": "parameter_name",
                    "current_value": "current_value",
                    "recommended_value": "new_value",
                    "priority": "high|medium|low",
                    "justification": "detailed justification",
                    "expected_impact": {
                        "performance": "expected performance impact",
                        "resource_usage": "expected resource impact",
                        "risks": ["potential_risk_1", "potential_risk_2"]
                    },
                    "implementation_steps": [
                        "step1",
                        "step2"
                    ]
                }
            ],
            "implementation_plan": {
                "order": ["param1", "param2"],
                "dependencies": ["dependency1", "dependency2"],
                "verification_steps": ["step1", "step2"]
            }
        }
        """

        self.config_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=config_template,
                input_variables=["problem_analysis", "current_config", "context"]
            )
        )

    def generate_recommendations(self, problem_analysis: Dict, current_config: Dict) -> Dict:
        """Generate configuration recommendations based on problem analysis."""
        # Generate specialized queries for configuration recommendations
        queries = [
            f"Hyperledger Fabric configuration recommendations for {problem['category']} problems: {problem['description']}"
            for problem in problem_analysis.get("problems", [])
        ]
        
        # Retrieve and combine relevant documents
        all_docs = []
        seen_docs = set()
        for query in queries:
            docs = self.retriever.get_relevant_documents(query)
            for doc in docs:
                doc_hash = hash(doc.page_content)
                if doc_hash not in seen_docs:
                    all_docs.append(doc)
                    seen_docs.add(doc_hash)

        # Combine retrieved contexts with metadata
        context = "\n\n".join([
            f"Reference {i+1} ({doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
            for i, doc in enumerate(all_docs[:5])
        ])

        # Run LLM chain for configuration recommendations
        result = self.config_chain.run(
            problem_analysis=json.dumps(problem_analysis, indent=2),
            current_config=json.dumps(current_config, indent=2),
            context=context
        )

        return json.loads(result)

class FabricParameterAdjustmentAgent:
    def __init__(self, llm: ChatOpenAI, vector_store_manager: VectorStoreManager):
        """Initialize the main agent with all required modules."""
        self.system_info_module = SystemInfoAcquisitionModule()
        self.problem_identification_module = ProblemIdentificationModule(llm, vector_store_manager)
        self.config_recommendation_module = ConfigurationRecommendationModule(llm, vector_store_manager)

    def analyze_and_recommend(self, raw_metrics: Dict) -> Dict:
        """Main workflow coordinating all modules."""
        try:
            # Step 1: Process system information
            metrics = self.system_info_module.extract_metrics(raw_metrics)

            # Step 2: Identify problems with enhanced analysis
            problem_analysis = self.problem_identification_module.analyze_problem(metrics)

            # Step 3: Generate detailed recommendations
            recommendations = self.config_recommendation_module.generate_recommendations(
                problem_analysis,
                metrics.current_config
            )

            # Combine results with metadata
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "analysis": {
                    "problem_analysis": problem_analysis,
                    "recommendations": recommendations,
                    "metrics_snapshot": {
                        "tps": metrics.tps,
                        "latency": metrics.latency,
                        "block_size": metrics.block_size,
                        "concurrent_requests": metrics.concurrent_requests,
                        "timeouts": metrics.timeouts
                    }
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__
            }

# Example usage
def main():
    # Initialize vector store manager and populate database
    vector_store_manager = VectorStoreManager(persist_directory="./chroma_db")
    
    # Initialize the database if it doesn't exist
    if not os.path.exists("../chroma_db"):
        vector_store_manager.initialize_knowledge_base()
    
    # Initialize the agent
    llm = ChatOpenAI(temperature=0)
    agent = FabricParameterAdjustmentAgent(llm, vector_store_manager)

    # Example metrics
    raw_metrics = {
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

    # Run analysis
    result = agent.analyze_and_recommend(raw_metrics)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()