from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import VectorStoreRetriever
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

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
    def __init__(self, llm: ChatOpenAI, knowledge_base_docs: List[str]):
        # Initialize embeddings and vector store for RAG
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_texts(
            knowledge_base_docs,
            self.embeddings
        )
        self.retriever = VectorStoreRetriever(
            vectorstore=self.vector_store
        )

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

        Relevant Knowledge Base Information:
        {context}

        Identify the main problem and its root cause. Format your response as JSON with the following structure:
        {
            "problem": "Brief description of the main problem",
            "root_cause": "Detailed explanation of the root cause",
            "severity": "high|medium|low"
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
        """Identify system problems and analyze root causes."""
        # Generate RAG query based on metrics
        query = f"Hyperledger Fabric performance issues with TPS: {metrics.tps}, Latency: {metrics.latency}"
        relevant_docs = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])

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
    def __init__(self, llm: ChatOpenAI, knowledge_base_docs: List[str]):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_texts(
            knowledge_base_docs,
            self.embeddings
        )
        self.retriever = VectorStoreRetriever(
            vectorstore=self.vector_store
        )

        config_template = """
        Based on the identified problem and current configuration, recommend parameter adjustments:

        Problem Information:
        {problem_info}

        Current Configuration:
        {current_config}

        Relevant Knowledge Base Information:
        {context}

        Generate configuration recommendations as JSON with the following structure:
        {
            "parameter_changes": [
                {
                    "parameter": "parameter_name",
                    "current_value": "current_value",
                    "recommended_value": "new_value",
                    "justification": "reason for change"
                }
            ],
            "expected_impact": "description of expected performance impact",
            "implementation_notes": "any special considerations for implementing changes"
        }
        """

        self.config_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=config_template,
                input_variables=["problem_info", "current_config", "context"]
            )
        )

    def generate_recommendations(self, problem_info: Dict, current_config: Dict) -> Dict:
        """Generate configuration recommendations based on problem analysis."""
        # Generate RAG query based on problem
        query = f"Hyperledger Fabric configuration recommendations for {problem_info['problem']}"
        relevant_docs = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Run LLM chain for configuration recommendations
        result = self.config_chain.run(
            problem_info=json.dumps(problem_info, indent=2),
            current_config=json.dumps(current_config, indent=2),
            context=context
        )

        return json.loads(result)

class FabricParameterAdjustmentAgent:
    def __init__(self, llm: ChatOpenAI, knowledge_base_docs: List[str]):
        """Initialize the main agent with all required modules."""
        self.system_info_module = SystemInfoAcquisitionModule()
        self.problem_identification_module = ProblemIdentificationModule(llm, knowledge_base_docs)
        self.config_recommendation_module = ConfigurationRecommendationModule(llm, knowledge_base_docs)

    def analyze_and_recommend(self, raw_metrics: Dict) -> Dict:
        """Main workflow coordinating all modules."""
        try:
            # Step 1: Process system information
            metrics = self.system_info_module.extract_metrics(raw_metrics)

            # Step 2: Identify problems
            problem_analysis = self.problem_identification_module.analyze_problem(metrics)

            # Step 3: Generate recommendations
            recommendations = self.config_recommendation_module.generate_recommendations(
                problem_analysis,
                metrics.current_config
            )

            # Combine results
            return {
                "status": "success",
                "problem_analysis": problem_analysis,
                "recommendations": recommendations
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Example usage
def main():
    # Initialize the agent
    llm = ChatOpenAI(temperature=0)
    knowledge_base_docs = [
        "Hyperledger Fabric best practices...",
        "Common configuration patterns...",
        # Add more knowledge base documents
    ]
    
    agent = FabricParameterAdjustmentAgent(llm, knowledge_base_docs)

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