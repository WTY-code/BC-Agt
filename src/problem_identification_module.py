# problem_identification_module.py
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

class ProblemIdentificationModule:
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
        self.retriever = vector_store_manager.get_retriever("problem_analysis")
        self.query_generator = RAGQueryGenerator(llm)
        
        self.template = """
        Analyze the following Hyperledger Fabric system metrics and identify potential problems:

        Performance Metrics:
        {performance_data}

        Current Configuration:
        {configuration_data}

        Retrieved Knowledge Base Information:
        {context}

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
            # performance_data = self.load_json_file(performance_path)
            # configuration_data = self.load_json_file(configuration_path)
            
            # queries = self.query_generator.generate_problem_analysis_queries(performance_path, configuration_path)
            queries = self.query_generator._generate_queries(
                "problem_analysis",
                performance_data=json.dumps(performance_data, indent=2),
                configuration_data=json.dumps(configuration_data, indent=2)
            )
            print("--------------queries---------------")
            print(queries)
            print("------------------------------------")
            
            all_docs = []
            seen_docs = set()
            for query_obj in queries:
                if not isinstance(query_obj, dict) or "query" not in query_obj:
                    print(f"Warning: Invalid query object format: {query_obj}")
                    continue
                    
                # 使用新的 invoke 方法替代 get_relevant_documents
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
            print("------------------------------------")

            prompt = PromptTemplate(
                template=self.template,
                input_variables=["performance_data", "configuration_data","context"]
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)

            analysis = chain.run({
                "performance_data": json.dumps(performance_data, indent=2),
                "configuration_data": json.dumps(configuration_data, indent=2),
                "context": context
            })

            return {
                "status": "success",
                "analysis": analysis
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
    module = ProblemIdentificationModule(llm, vector_store_manager)
    
    result = module.process(
        "./input/performance.json",
        "./input/configuration.json"
    )

    print("---------------analysis-----------------")
    print(json.dumps(result, indent=2))