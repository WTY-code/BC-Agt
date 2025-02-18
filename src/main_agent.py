# main_agent.py
from datetime import datetime
from typing import Dict
import json
import os

from langchain.chat_models import ChatOpenAI
from vector_store import VectorStoreManager
from system_info_acquisition_module import SystemInfoAcquisitionModule
from problem_identification_module import ProblemIdentificationModule
from configuration_recommendation_module import ConfigurationRecommendationModule

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
            metrics_result = self.system_info_module.process(raw_metrics)
            if metrics_result["status"] == "error":
                return metrics_result

            # Step 2: Identify problems with enhanced analysis
            problem_result = self.problem_identification_module.process(metrics_result["metrics"])
            if problem_result["status"] == "error":
                return problem_result

            # Step 3: Generate detailed recommendations
            recommendation_input = {
                "problem_analysis": problem_result["analysis"],
                "current_config": raw_metrics["current_config"]
            }
            recommendation_result = self.config_recommendation_module.process(recommendation_input)
            if recommendation_result["status"] == "error":
                return recommendation_result

            # Combine results with metadata
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "analysis": {
                    "problem_analysis": problem_result["analysis"],
                    "recommendations": recommendation_result["recommendations"],
                    "metrics_snapshot": metrics_result["metrics"]
                }
            }

        # main_agent.py (continued)

        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__
            }

def main():
    """Example usage of the Fabric Parameter Adjustment Agent."""
    try:
        # Initialize vector store manager and populate database
        vector_store_manager = VectorStoreManager(persist_directory="./chroma_db")
        
        # Initialize the database if it doesn't exist
        if not os.path.exists("./chroma_db"):
            print("Initializing knowledge base...")
            vector_store_manager.initialize_knowledge_base()
        
        # Initialize the agent
        print("Initializing LLM agent...")
        llm = ChatOpenAI(temperature=0)
        agent = FabricParameterAdjustmentAgent(llm, vector_store_manager)

        # Example metrics
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

        # Run analysis
        print("Running analysis...")
        result = agent.analyze_and_recommend(example_metrics)
        
        # Print results
        print("\nAnalysis Results:")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()