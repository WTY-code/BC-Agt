import os
from openai import OpenAI  # 新版本导入方式
from dotenv import load_dotenv
import requests
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_proxy_settings():
    """验证代理设置"""
    http_proxy = os.getenv('HTTP_PROXY')
    https_proxy = os.getenv('HTTPS_PROXY')
    
    logger.info("Current proxy settings:")
    logger.info(f"HTTP_PROXY: {http_proxy}")
    logger.info(f"HTTPS_PROXY: {https_proxy}")
    
    if not (http_proxy or https_proxy):
        logger.warning("⚠️ No proxy settings found!")
        return False
    
    # 测试代理连接
    try:
        proxies = {
            'http': http_proxy,
            'https': https_proxy
        }
        
        response = requests.get(
            'http://httpbin.org/ip',
            proxies=proxies,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Proxy test successful. IP: {response.json().get('origin')}")
            return True
        else:
            logger.error(f"Proxy test failed with status code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Proxy test failed: {str(e)}")
        return False

def test_api_connection():
    """测试与 OpenAI API 的连接"""
    try:
        # 初始化 OpenAI 客户端
        client = OpenAI()
        
        # 测试模型列表访问
        models = client.models.list()
        logger.info("✅ Successfully connected to OpenAI API")
        logger.info(f"Available models: {len(models.data)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ API connection failed: {str(e)}")
        return False

def test_embeddings():
    """测试嵌入生成"""
    try:
        client = OpenAI()
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="This is a test message.",
            encoding_format="float"
        )
        
        logger.info("✅ Successfully generated embeddings")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embeddings generation failed: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("Starting OpenAI API validation...")
    logger.info("-" * 50)
    
    # 加载环境变量
    load_dotenv()
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 验证 API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not found in environment variables")
        return
    logger.info("✅ API key found")
    
    # 验证代理设置
    proxy_status = verify_proxy_settings()
    if not proxy_status:
        logger.warning("Consider checking your proxy configuration")
    
    # 运行测试
    tests = {
        "API Connection": test_api_connection,
        "Embeddings Generation": test_embeddings
    }
    
    results = {}
    for test_name, test_func in tests.items():
        logger.info(f"\nRunning test: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}")
            results[test_name] = False
    
    # 打印总结
    logger.info("\n" + "=" * 50)
    logger.info("Test Summary:")
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    duration = datetime.now() - start_time
    logger.info("\n" + "=" * 50)
    logger.info(f"Total time: {duration.total_seconds():.2f} seconds")
    logger.info(f"Overall status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()