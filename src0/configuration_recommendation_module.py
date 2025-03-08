# configuration_recommendation_module.py
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Dict
from dotenv import load_dotenv
import json
import os

load_dotenv()
api_key = os.getenv('LINKAI_API_KEY')
api_base = os.getenv('LINKAI_API_BASE')

class ConfigurationRecommendationModule:
    def __init__(self, llm=None):
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
        
        self.template = """
        Based on the identified problems and current configuration, recommend parameter adjustments:

        Problem Analysis:
        {problem_analysis}

        Current Configuration:
        {configuration_data}

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
            
            # If analysis is already a dict, return it directly
            if isinstance(analysis_result["analysis"], dict):
                return analysis_result["analysis"]
            
            # If analysis is a string (possibly in markdown format), try to parse it
            if isinstance(analysis_result["analysis"], str):
                analysis_str = analysis_result["analysis"]
                analysis_str = analysis_str.replace('```json\n', '').replace('\n```', '').strip('`').strip()
                return json.loads(analysis_str)
            
            raise ValueError("Unexpected analysis format")
        except Exception as e:
            raise Exception(f"Error parsing problem analysis: {str(e)}")

    def generate_recommendations(self, analysis_result: Dict, configuration_data: Dict) -> Dict:
        """Generate configuration recommendations based on problem analysis."""
        try:
            # Parse problem analysis
            problem_analysis = self.parse_problem_analysis(analysis_result)
            
            # Create prompt template and chain
            prompt = PromptTemplate(
                template=self.template,
                input_variables=["problem_analysis", "configuration_data"]
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)

            # Generate recommendations
            result = chain.run({
                "problem_analysis": json.dumps(problem_analysis, indent=2),
                "configuration_data": json.dumps(configuration_data, indent=2)
            })

            # Clean and parse the result
            cleaned_result = result.replace('```json\n', '').replace('```\n', '').strip('`').strip()
            
            try:
                recommendations = json.loads(cleaned_result)
                return {
                    "status": "success",
                    "recommendations": recommendations
                }
            except json.JSONDecodeError:
                # If parsing fails, return the raw string
                return {
                    "status": "success",
                    "recommendations": result
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
        
    def process(self, analysis_result: Dict, configuration_path: str) -> Dict:
        """Generate configuration recommendations based on problem analysis."""
        configuration_data = self.load_json_file(configuration_path)

        return self.generate_recommendations(analysis_result, configuration_data)
    
if __name__ == "__main__":
    # Example usage
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
    
    module = ConfigurationRecommendationModule(llm)
    
    # Example problem analysis result from problem identification module
    example_analysis_result0 = {
        "status": "success",
        "analysis": {
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
    }

    example_analysis_result = {
        "status": "success",
        "analysis": {
            "problems": [
                {
                    "category": "performance",
                    "description": "High latency in 'Create a car' and 'Change car owner' transactions, with average latencies of 1.17s and 1.50s respectively, and maximum latencies of 2.12s and 2.04s.",
                    "severity": "high",
                    "impact": "Delays in transaction processing can lead to poor user experience and reduced system efficiency.",
                    "related_metrics": [
                    "MaxLatency",
                    "AvgLatency"
                    ]
                },
                {
                    "category": "configuration",
                    "description": "BatchTimeout is set to 2s, which may be too short for high-latency transactions, causing frequent batching and potential delays.",
                    "severity": "medium",
                    "impact": "Frequent batching can lead to inefficiencies in transaction processing and increased latency.",
                    "related_metrics": [
                    "BatchTimeout"
                    ]
                },
                {
                    "category": "resource",
                    "description": "CPU usage shows a double-spike phenomenon with peaks at 77.8% and 68%, indicating potential resource contention or inefficient processing.",
                    "severity": "medium",
                    "impact": "High CPU usage can lead to system instability and reduced performance.",
                    "related_metrics": [
                    "cpu_analysis"
                    ]
                },
                {
                    "category": "architecture",
                    "description": "Single orderer node (orderer.example.com) may create a single point of failure and limit scalability.",
                    "severity": "medium",
                    "impact": "Single point of failure can lead to system downtime, and limited scalability can hinder performance under high load.",
                    "related_metrics": [
                    "Orderer"
                    ]
                }
            ],
            "root_causes": [
                {
                    "problem_ref": 0,
                    "description": "High latency in 'Create a car' and 'Change car owner' transactions is likely due to inefficient transaction processing and potential network delays.",
                    "confidence": "high",
                    "evidence": "Metrics show high average and maximum latencies for these transactions."
                },
                {
                    "problem_ref": 1,
                    "description": "BatchTimeout of 2s may be too short for high-latency transactions, causing frequent batching and delays.",
                    "confidence": "medium",
                    "evidence": "Configuration shows BatchTimeout set to 2s, which may not be optimal for high-latency transactions."
                },
                {
                    "problem_ref": 2,
                    "description": "CPU spikes may be caused by inefficient processing or resource contention during high transaction loads.",
                    "confidence": "medium",
                    "evidence": "CPU analysis shows double-spike phenomenon with high peaks."
                },
                {
                    "problem_ref": 3,
                    "description": "Single orderer node can create a single point of failure and limit scalability, especially under high transaction loads.",
                    "confidence": "high",
                    "evidence": "Configuration shows only one orderer node, which may not be sufficient for high scalability and fault tolerance."
                }
            ]
        }
    }
    
    result = module.process(
        analysis_result=example_analysis_result,
        configuration_path="./input/configuration.json"
    )
    
    print("---------------recommendations-----------------")
    print(json.dumps(result, indent=2))