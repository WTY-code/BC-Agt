# rag_generation.py
from typing import Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os, json

load_dotenv()
api_key = os.getenv('LINKAI_API_KEY')
api_base = os.getenv('LINKAI_API_BASE')

class RAGQueryGenerator:
    def __init__(self, llm: ChatOpenAI = None):
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
        
        # Define query generation templates
        self.templates = {
            "problem_analysis": """
                Given the following Hyperledger Fabric system metrics and configuration, generate search queries that will help identify and analyze potential problems.

                Performance Metrics:
                {performance_data}

                Configuration:
                {configuration_data}

                Your task is to generate search queries that will help retrieve documents relevant to:
                1. Performance bottleneck identification
                2. Resource utilization issues
                3. Configuration-related problems
                4. System architecture concerns
                5. Common failure patterns and their symptoms

                Focus on queries that will help understand:
                - What could be causing the current performance patterns
                - How the current configuration might be contributing to issues
                - Typical problem patterns in similar setups
                - Common failure modes and their indicators

                Return the queries in JSON format as shown in this example:
                {{
                    "queries": [
                        {{
                            "query": "Hyperledger Fabric performance bottlenecks in endorsement policy",
                            "purpose": "Identify common bottlenecks in transaction flow",
                            "relevance": "high"
                        }}
                    ]
                }}

                Generate at least 3 relevant queries.""",
            
            "recommendation": """
                Based on the following problem analysis and current configuration, generate search queries for configuration recommendations:

                Problem Analysis:
                {problem_analysis}

                Current Configuration:
                {configuration_data}

                Your task is to generate search queries that will help retrieve documents relevant to optimizing the configuration to address the identified problems. Consider:
                1. Configuration solutions for each identified problem
                2. Best practices for the affected components
                3. Performance tuning patterns for similar issues
                4. Risk mitigation strategies
                5. Dependencies between configuration parameters

                Focus on queries that will help find:
                - Specific configuration parameters that need adjustment
                - Proven optimization strategies for similar problems
                - Trade-offs and potential impacts of changes
                - Implementation considerations and prerequisites
                - Validation and testing approaches

                Return the queries in JSON format as shown in this example:
                {{
                    "queries": [
                        {{
                            "query": "Hyperledger Fabric endorsement policy optimization for high latency",
                            "purpose": "Find configuration patterns that reduce endorsement latency",
                            "relevance": "high"
                        }}
                    ]
                }}

                Generate at least 3 relevant queries, prioritizing the most severe problems."""
        }
        
        # Fallback queries for error cases
        self.fallback_queries = {
            "problem_analysis": {
                "query": "Hyperledger Fabric common performance problems and diagnosis",
                "purpose": "General problem identification",
                "relevance": "high"
            },
            "recommendation": {
                "query": "Hyperledger Fabric configuration best practices and optimization",
                "purpose": "General configuration recommendations",
                "relevance": "high"
            }
        }

    def load_json_file(self, file_path: str) -> Dict:
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {str(e)}")

    def _generate_queries(self, template_name: str, **kwargs) -> List[Dict]:
        """Common function for generating queries using specified template."""
        try:
            # Create prompt template and chain dynamically
            prompt = PromptTemplate(
                template=self.templates[template_name],
                input_variables=list(kwargs.keys())
            )
            chain = LLMChain(llm=self.llm, prompt=prompt)

            # Generate queries
            result = chain.run(kwargs)
            
            try:
                # Clean and parse the returned JSON
                cleaned_result = result.replace('```json\n', '').replace('```\n', '').strip('`').strip()
                query_data = json.loads(cleaned_result)
                
                if "queries" in query_data:
                    return query_data["queries"]
                else:
                    print(f"Warning: Unexpected response format. Using fallback query.")
                    return [self.fallback_queries[template_name]]
            except json.JSONDecodeError as e:
                print(f"Error parsing response as JSON: {str(e)}")
                print("Raw response:", result)
                return [self.fallback_queries[template_name]]

        except Exception as e:
            print(f"Error generating {template_name} queries: {str(e)}")
            return [self.fallback_queries[template_name]]

    def generate_problem_analysis_queries(self, performance_path: str, configuration_path: str) -> List[Dict]:
        """Generate queries focused on problem analysis and identification."""
        performance_data = self.load_json_file(performance_path)
        configuration_data = self.load_json_file(configuration_path)
        
        return self._generate_queries(
            "problem_analysis",
            performance_data=json.dumps(performance_data, indent=2),
            configuration_data=json.dumps(configuration_data, indent=2)
        )

    def generate_recommendation_queries(self, problem_analysis: Dict, configuration_path: str) -> List[Dict]:
        """Generate queries focused on configuration recommendations based on problem analysis."""
        configuration_data = self.load_json_file(configuration_path)
        
        return self._generate_queries(
            "recommendation",
            problem_analysis=json.dumps(problem_analysis, indent=2),
            configuration_data=json.dumps(configuration_data, indent=2)
        )

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
    
    generator = RAGQueryGenerator(llm)
    
    # Test problem analysis query generation
    problem_queries = generator.generate_problem_analysis_queries(
        "./input/performance.json",
        "./input/configuration.json"
    )
    print("\nProblem Analysis Queries:")
    print(json.dumps(problem_queries, indent=2))
    
    # Test recommendation query generation with example problem analysis
    example_problem_analysis = {
        "problems": [
            {
                "category": "performance",
                "description": "High transaction latency in endorsement phase",
                "severity": "high",
                "impact": "Significant delay in transaction processing",
                "related_metrics": ["endorsement_latency", "transaction_throughput"]
            }
        ],
        "root_causes": [
            {
                "problem_ref": 0,
                "description": "Endorsement policy requires too many endorsers",
                "confidence": "high",
                "evidence": "Current endorsement policy requires signatures from all organizations"
            }
        ]
    }
    
    recommendation_queries = generator.generate_recommendation_queries(
        example_problem_analysis,
        "./input/configuration.json"
    )
    print("\nRecommendation Queries:")
    print(json.dumps(recommendation_queries, indent=2))