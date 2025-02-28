# main_agent.py
from datetime import datetime
from typing import Dict, Any
import json
import os
from langchain.chat_models import ChatOpenAI
from vector_store import VectorStoreManager
from problem_identification_module import ProblemIdentificationModule
from configuration_recommendation_module import ConfigurationRecommendationModule
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('LINKAI_API_KEY')
api_base = os.getenv('LINKAI_API_BASE')

class FabricParameterAdjustmentAgent:
    def __init__(self, llm=None, vector_store_manager=None):
        """Initialize the main agent with all required modules."""
        # Initialize LLM if not provided
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
        
        # Initialize vector store manager if not provided
        if not vector_store_manager:
            self.vector_store_manager = VectorStoreManager(persist_directory="./chroma_db")
            if not os.path.exists("./chroma_db"):
                print("Initializing knowledge base...")
                self.vector_store_manager.initialize_knowledge_base()
        else:
            self.vector_store_manager = vector_store_manager
        
        # Initialize modules
        self.problem_identification_module = ProblemIdentificationModule(self.llm, self.vector_store_manager)
        self.config_recommendation_module = ConfigurationRecommendationModule(self.llm, self.vector_store_manager)
        
        # Create output directory if it doesn't exist
        # os.makedirs("./input", exist_ok=True)
        os.makedirs("./output", exist_ok=True)

    def load_json_file(self, file_path: str) -> Dict:
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {str(e)}")
    
    def save_json_file(self, data: Dict, file_path: str) -> None:
        """Save data to a JSON file."""
        try:
            with open(file_path, 'w') as file:
                json.dump(data, indent=2, fp=file)
            print(f"Successfully saved to {file_path}")
        except Exception as e:
            raise Exception(f"Error saving to {file_path}: {str(e)}")

    def process(self, performance_path: str, configuration_path: str, output_path: str = "./output/result.json") -> Dict:
        """
        Main workflow coordinating all modules.
        
        Args:
            performance_path: Path to the performance metrics JSON file
            configuration_path: Path to the configuration JSON file
            output_path: Path to save the final output
            
        Returns:
            Dict containing the analysis results
        """
        try:
            timestamp = datetime.now().isoformat()
            print(f"[{timestamp}] Starting analysis workflow")
            
            # Ensure the input files exist
            print("Validating input files...")
            if not os.path.exists(performance_path):
                return {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": f"Performance metrics file not found: {performance_path}"
                }
                
            if not os.path.exists(configuration_path):
                return {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": f"Configuration file not found: {configuration_path}"
                }
            
            # Copy input files to the standard location for other modules
            print("Preparing input files...")
            performance_data = self.load_json_file(performance_path)
            configuration_data = self.load_json_file(configuration_path)
            
            # Save these to the input directory for other modules to use
            # self.save_json_file(performance_data, "./input/performance.json")
            # self.save_json_file(configuration_data, "./input/configuration.json")
            
            # Step 2: Problem identification
            print("Analyzing problems...")
            problem_analysis = self.problem_identification_module.analyze_problem(
                performance_data,
                configuration_data
            )
            
            if problem_analysis["status"] != "success":
                print(f"Error in problem identification: {problem_analysis.get('error', 'Unknown error')}")
                return problem_analysis
                
            # Save intermediate result
            self.save_json_file(problem_analysis, "./output/problem_analysis.json")
            print("Problem analysis completed successfully")
            
            # Step 3: Generate configuration recommendations
            print("Generating configuration recommendations...")
            try:
                # Parse problem analysis result for the recommendation module
                # problem_analysis = self.problem_identification_module.parse_problem_analysis(problem_result)
                
                recommendation_result = self.config_recommendation_module.generate_recommendations(
                    problem_analysis,
                    configuration_data
                )
                
                if recommendation_result["status"] != "success":
                    print(f"Error in configuration recommendation: {recommendation_result.get('error', 'Unknown error')}")
                    return recommendation_result
                    
                # Save intermediate result
                self.save_json_file(recommendation_result, "./output/recommendations.json")
                print("Configuration recommendations generated successfully")
                
            except Exception as e:
                recommendation_error = {
                    "status": "error",
                    "error": f"Error generating recommendations: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                self.save_json_file(recommendation_error, "./output/recommendations.json")
                return recommendation_error
            
            # Step 4: Combine results
            final_result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "problem_analysis": problem_analysis,
                "recommendations": recommendation_result,
                "input": {
                    "performance_path": performance_path,
                    "configuration_path": configuration_path
                }
            }
            
            # Save final result
            self.save_json_file(final_result, output_path)
            print(f"Analysis workflow completed successfully. Results saved to {output_path}")
            
            return final_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__
            }
            self.save_json_file(error_result, output_path)
            print(f"Error in analysis workflow: {str(e)}")
            return error_result

def main():
    """Example usage of the Fabric Parameter Adjustment Agent."""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Hyperledger Fabric Parameter Adjustment Agent')
        parser.add_argument('--performance', default='./input/performance.json', 
                            help='Path to performance metrics JSON file')
        parser.add_argument('--configuration', default='./input/configuration.json', 
                            help='Path to configuration JSON file')
        parser.add_argument('--output', default='./output/result.json',
                            help='Path to save the output JSON file')
        args = parser.parse_args()
        
        # Initialize the agent
        print("Initializing LLM agent...")
        agent = FabricParameterAdjustmentAgent()
        
        # Run analysis
        print(f"Running analysis with:\n- Performance: {args.performance}\n- Configuration: {args.configuration}")
        result = agent.process(args.performance, args.configuration, args.output)
        
        # Print summary
        if result["status"] == "success":
            print("\nAnalysis completed successfully.")
            
            if "problem_analysis" in result and "analysis" in result["problem_analysis"]:
                print("\nProblem Analysis:")
                print(result["problem_analysis"]["analysis"][:500] + "..." 
                      if len(result["problem_analysis"]["analysis"]) > 500 
                      else result["problem_analysis"]["analysis"])
                
            if "recommendations" in result and "recommendations" in result["recommendations"]:
                print("\nRecommendations Summary:")
                recommendations = result["recommendations"]["recommendations"]
                if isinstance(recommendations, dict) and "recommendations" in recommendations:
                    for i, rec in enumerate(recommendations["recommendations"]):
                        print(f"  {i+1}. {rec.get('parameter')}: {rec.get('current_value')} → {rec.get('recommended_value')}")
                        print(f"     Priority: {rec.get('priority')}")
                        print(f"     Justification: {rec.get('justification')[:100]}..." 
                              if len(rec.get('justification', '')) > 100 
                              else rec.get('justification', ''))
        else:
            print("\nAnalysis failed:")
            print(f"Error: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()