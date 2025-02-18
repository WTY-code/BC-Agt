# configuration_recommendation_module.py
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from vector_store import VectorStoreManager
from rag_generation import RAGQueryGenerator
from typing import List, Dict
import json
import os

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
        queries = [
            f"Hyperledger Fabric configuration recommendations for {problem['category']} problems: {problem['description']}"
            for problem in problem_analysis.get("problems", [])
        ]
        
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
            f"Reference {i+1} ({doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
            for i, doc in enumerate(all_docs[:5])
        ])

        result = self.config_chain.run(
            problem_analysis=json.dumps(problem_analysis, indent=2),
            current_config=json.dumps(current_config, indent=2),
            context=context
        )

        return json.loads(result)

    def process(self, input_data: Dict) -> Dict:
        """Process problem analysis and current configuration to generate recommendations."""
        try:
            recommendations = self.generate_recommendations(
                input_data["problem_analysis"],
                input_data["current_config"]
            )
            return {
                "status": "success",
                "recommendations": recommendations
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
    module = ConfigurationRecommendationModule(llm, vector_store_manager)
    
    example_input = {
        "problem_analysis": {
            "problems": [
                {
                    "category": "performance",
                    "description": "Low transaction throughput",
                    "severity": "high",
                    "impact": "System cannot handle required transaction load",
                    "related_metrics": ["tps", "latency"]
                }
            ]
        },
        "current_config": {
            "block_size": 10,
            "consensus_type": "kafka",
            "endorsement_policy": "AND('Org1.member', 'Org2.member')"
        }
    }
    
    result = module.process(example_input)
    print(json.dumps(result, indent=2))