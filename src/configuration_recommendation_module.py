# configuration_recommendation_module.py
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from vector_store import VectorStoreManager
from rag_generation import RAGQueryGenerator
from typing import List, Dict
from dotenv import load_dotenv
import json
import os

load_dotenv()
api_key = os.getenv('LINKAI_API_KEY')
api_base = os.getenv('LINKAI_API_BASE')

class ConfigurationRecommendationModule:
    def __init__(self, llm, vector_store_manager: VectorStoreManager):
        self.llm = llm if llm else ChatOpenAI(
            temperature=0,
            model_name="deepseek-chat",                
            openai_api_base=api_base,
            openai_api_key=api_key,  
            max_tokens=None,            
            streaming=False,            
            request_timeout=None,       
            max_retries=6,             
            model_kwargs={},
        )
        self.retriever = vector_store_manager.get_retriever("config_recommendation")
        self.query_generator = RAGQueryGenerator(llm)
        
        self.template = """
        Based on the identified problems and current configuration, recommend parameter adjustments:

        Problem Analysis:
        {problem_analysis}

        Current Configuration:
        {configuration_data}

        Retrieved Knowledge Base Information:
        {context}

        Based on the above information, provide a comprehensive configuration recommendation following these aspects:
        1. Impact on system performance
        2. Implementation complexity
        3. Resource requirements
        4. Potential risks and mitigation strategies
        5. Dependencies between parameters

        Format your response as JSON with the following structure:
        {{
            "recommendations": [
                {{
                    "parameter": "parameter_name",
                    "current_value": "current_value",
                    "recommended_value": "new_value",
                    "priority": "high|medium|low",
                    "justification": "detailed justification",
                    "expected_impact": {{
                        "performance": "expected performance impact",
                        "resource_usage": "expected resource impact",
                        "risks": ["potential_risk_1", "potential_risk_2"]
                    }},
                    "implementation_steps": [
                        "step1",
                        "step2"
                    ]
                }}
            ],
            "implementation_plan": {{
                "order": ["param1", "param2"],
                "dependencies": ["dependency1", "dependency2"],
                "verification_steps": ["step1", "step2"]
            }}
        }}
        """

    def load_json_file(self, file_path: str) -> Dict:
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {str(e)}")

    def parse_problem_analysis(self, analysis_result: Dict) -> Dict:
        """Parse the problem analysis result from the problem identification module."""
        try:
            if "analysis" not in analysis_result or analysis_result["status"] != "success":
                raise ValueError("Invalid analysis result format")
            
            # Clean the JSON string from markdown format
            analysis_str = analysis_result["analysis"]
            analysis_str = analysis_str.replace('```json\n', '').replace('\n```', '').strip('`').strip()
            
            return json.loads(analysis_str)
        except Exception as e:
            raise Exception(f"Error parsing problem analysis: {str(e)}")

    def generate_recommendations(self, analysis_result: Dict, configuration_path: str) -> Dict:
        """Generate configuration recommendations based on problem analysis."""
        try:
            # Parse problem analysis
            problem_analysis = self.parse_problem_analysis(analysis_result)
            
            # Load configuration data
            configuration_data = self.load_json_file(configuration_path)
            
            # Generate queries based on identified problems
            queries = self.query_generator.generate_recommendation_queries(
                problem_analysis=problem_analysis,
                configuration_path=configuration_path
            )
            
            print("--------------queries---------------")
            print(queries)
            print("------------------------------------")
            
            # Collect relevant documents
            all_docs = []
            seen_docs = set()
            for query_obj in queries:
                if not isinstance(query_obj, dict) or "query" not in query_obj:
                    print(f"Warning: Invalid query object format: {query_obj}")
                    continue
                
                docs = self.retriever.invoke(query_obj["query"])
                
                for doc in docs:
                    doc_hash = hash(doc.page_content)
                    if doc_hash not in seen_docs:
                        all_docs.append(doc)
                        seen_docs.add(doc_hash)

            context = "\n\n".join([
                f"Reference {i+1}:\n{doc.page_content}\n\nSource: {doc.metadata.get('source', 'Unknown')}"
                for i, doc in enumerate(all_docs[:5])
            ])

            print("---------------context-----------------")
            print(context)
            print("---------------------------------------")

            # Create prompt template and chain
            prompt = PromptTemplate(
                template=self.template,
                input_variables=["problem_analysis", "configuration_data", "context"]
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)

            # Generate recommendations
            result = chain.run({
                "problem_analysis": json.dumps(problem_analysis, indent=2),
                "configuration_data": json.dumps(configuration_data, indent=2),
                "context": context
            })

            # Clean and parse the result
            cleaned_result = result.replace('```json\n', '').replace('```\n', '').strip('`').strip()
            recommendations = json.loads(cleaned_result)

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
    vector_store_manager = VectorStoreManager(persist_directory="./chroma_db")
    if not os.path.exists("./chroma_db"):
        vector_store_manager.initialize_knowledge_base()
    
    llm = ChatOpenAI(
        temperature=0,
        model_name="deepseek-chat",                
        openai_api_base=api_base,
        openai_api_key=api_key,  
        max_tokens=None,            
        streaming=False,            
        request_timeout=None,       
        max_retries=6,             
        model_kwargs={},
    )
    
    module = ConfigurationRecommendationModule(llm, vector_store_manager)
    
    # Example problem analysis result from problem identification module
    example_analysis_result = {
        "status": "success",
        "analysis": """```json
        {
            "problems": [
                {
                    "category": "performance",
                    "description": "High average latency of 2.3s and maximum latency of 4.1s",
                    "severity": "high",
                    "impact": "Increased transaction processing time",
                    "related_metrics": ["AvgLatency", "MaxLatency"]
                }
            ],
            "root_causes": [
                {
                    "problem_ref": 0,
                    "description": "High latency due to network delays or inefficient chaincode execution",
                    "confidence": "medium",
                    "evidence": "AvgLatency: 2.3s, MaxLatency: 4.1s"
                }
            ]
        }
        ```"""
    }
    
    result = module.generate_recommendations(
        analysis_result=example_analysis_result,
        configuration_path="./input/configuration.json"
    )
    
    print("---------------recommendations-----------------")
    print(json.dumps(result, indent=2))