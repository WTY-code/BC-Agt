from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 创建 LLM 实例
llm = ChatOpenAI(
    temperature=0,
    # model_name="LinkAI-3.5",
    # model_name="LinkAI-4o",
    # model_name="claude-3-sonnet",
    # model_name="qwen-plus",
    # model_name="deepseek-chat", 
    model_name="deepseek-reasoner",              
    openai_api_key="LINKAI_API_KEY",    
    openai_api_base="https://api.link-ai.tech/v1",  
    max_tokens=None,            
    streaming=False,            
    request_timeout=None,       
    max_retries=6,             
    model_kwargs={},           
)

# 创建 PromptTemplate
prompt_template = PromptTemplate(
    input_variables=[],  # 如果prompt中没有需要填充的变量，使用空列表
    template="""
    Please tell me about yourself:
    1. Which company developed your language model?
    2. What is the name of your model?
    3. Which version are you using?   
    Please answer the above questions directly without any additional statements or explanations."""
)

# 创建并运行 chain
chain = LLMChain(llm=llm, prompt=prompt_template)
result = chain.run({})
# result = chain.invoke({})
print(result)