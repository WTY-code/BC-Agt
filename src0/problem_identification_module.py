# problem_identification_module.py
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

class ProblemIdentificationModule:
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
        Analyze the following Hyperledger Fabric system metrics and identify potential problems:

        Performance Metrics:
        {performance_data}

        Current Configuration:
        {configuration_data}

        Based on the above information, provide a comprehensive analysis following these steps:
        1. Identify performance bottlenecks
        2. Analyze configuration issues
        3. Evaluate resource utilization
        4. Consider system architecture implications

        Format your response as JSON with the following structure:
        {{
            "problems": [
                {{
                    "category": "performance|configuration|resource|architecture",
                    "description": "Detailed problem description",
                    "severity": "high|medium|low",
                    "impact": "Impact on system performance",
                    "related_metrics": ["affected_metric1", "affected_metric2"]
                }}
            ],
            "root_causes": [
                {{
                    "problem_ref": "Index of related problem",
                    "description": "Detailed root cause explanation",
                    "confidence": "high|medium|low",
                    "evidence": "Evidence from metrics and configuration"
                }}
            ]
        }}
        """

    def load_json_file(self, file_path: str) -> Dict:
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {str(e)}")

    def analyze_problem(self, performance_data: Dict, configuration_data: Dict) -> Dict:
        """Process metrics and return problem analysis."""
        try:
            prompt = PromptTemplate(
                template=self.template,
                input_variables=["performance_data", "configuration_data"]
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)

            analysis = chain.run({
                "performance_data": json.dumps(performance_data, indent=2),
                "configuration_data": json.dumps(configuration_data, indent=2)
            })

            # Clean and parse the analysis result
            cleaned_analysis = analysis.replace('```json\n', '').replace('\n```', '').strip('`').strip()
            parsed_analysis = json.loads(cleaned_analysis)

            return {
                "status": "success",
                "analysis": parsed_analysis
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
        
    def process(self, performance_path: str, configuration_path: str) -> Dict:
        """Process metrics and return problem analysis."""
        performance_data = self.load_json_file(performance_path)
        configuration_data = self.load_json_file(configuration_path)
        
        return self.analyze_problem(performance_data, configuration_data)

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
    module = ProblemIdentificationModule(llm)
    
    result = module.process(
        "./input/performance.json",
        "./input/configuration.json"
    )

    print("---------------analysis-----------------")
    print(json.dumps(result, indent=2))