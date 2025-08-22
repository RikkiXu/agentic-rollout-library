#!/usr/bin/env python3
"""
LLM API Client for agentic-rollout-library.

Provides a unified interface for calling various LLM APIs using OpenAI SDK compatibility.
"""

import asyncio
import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LLMAPIClient:
    """LLM API客户端 - 使用OpenAI SDK接口兼容多种LLM提供商"""
    
    def __init__(self, api_key: str, base_url: str, model: str, debug: bool = False, max_retries: int = 3,
                 proxy_url: Optional[str] = None):
        """
        Initialize LLM API client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL of the API endpoint
            model: Model name to use for generation
            debug: Enable debug mode to log all API inputs and outputs
            max_retries: Maximum number of retry attempts on failure (default: 3)
            proxy_url: Optional proxy URL to use for requests (overrides environment variables)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.debug = debug
        self.max_retries = max_retries
        
        try:
            import openai
            import httpx
            
            # 优先使用显式传入的代理配置，否则检测环境变量
            if proxy_url:
                http_proxy = proxy_url
                https_proxy = proxy_url
            else:
                # 检测代理环境变量
                http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
                https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
            
            # 如果有代理配置，创建自定义的 httpx 客户端
            if http_proxy or https_proxy:
                # httpx 使用统一的代理配置，优先使用 https_proxy
                proxy_url = https_proxy or http_proxy
                logger.info(f"使用代理: {proxy_url}")
                
                # 创建配置了代理的 httpx 客户端
                # 重要：trust_env=False 确保不会受环境变量变化影响
                http_client = httpx.AsyncClient(
                    proxy=proxy_url,  # 使用 proxy 参数，而不是 proxies
                    timeout=httpx.Timeout(120.0, connect=30.0),  # 增加超时时间
                    verify=True,  # 验证 SSL 证书
                    follow_redirects=True,
                    trust_env=False  # 禁用自动读取环境变量，避免被 kodo 等工具干扰
                )
                
                # 使用自定义 http 客户端创建 OpenAI 客户端
                self.client = openai.AsyncOpenAI(
                    api_key=api_key,
                    base_url=f"{base_url}",
                    timeout=60.0,
                    http_client=http_client
                )
                logger.info(f"OpenAI 客户端已配置代理访问")
            else:
                # 没有代理配置，使用默认设置
                self.client = openai.AsyncOpenAI(
                    api_key=api_key,
                    base_url=f"{base_url}",
                    timeout=120.0  # 增加超时时间到 120 秒
                )
                logger.debug("OpenAI 客户端使用直连模式（无代理）")
                
        except ImportError as e:
            if "openai" in str(e):
                logger.error("需要安装openai库: pip install openai")
            elif "httpx" in str(e):
                logger.error("需要安装httpx库: pip install httpx")
            raise
    
    async def generate(self, messages: List[Dict[str, str]], max_tokens: int = 16000, temperature: float = 0.7) -> str:
        """
        Call LLM API to generate response with retry mechanism.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        if self.debug:
            self._log_debug_request(messages)
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    # 计算重试延迟 (指数退避: 1, 2, 4秒)
                    delay = 2 ** (attempt - 1)
                    logger.warning(f"🔄 重试第 {attempt} 次，延迟 {delay} 秒...")
                    await asyncio.sleep(delay)
                
                
                # 执行API调用
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # 处理不同的响应格式
                content = self._extract_response_content(response)
                
                # Debug模式：记录API响应详情
                if self.debug:
                    self._log_debug_response(response, content, attempt)
                
                return content
                
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                
                # 判断是否应该重试
                should_retry = self._should_retry_on_error(e, attempt)
                
                if should_retry and attempt < self.max_retries:
                    logger.warning(f"❌ API调用失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                    logger.warning(f"   将在延迟后重试...")
                else:
                    # 最后一次尝试失败或不应重试的错误
                    if attempt >= self.max_retries:
                        logger.error(f"❌ API调用最终失败，已达到最大重试次数 ({self.max_retries + 1})")
                    else:
                        logger.error(f"❌ API调用失败，错误类型不支持重试: {error_msg}")
                    
                    raise Exception(f"LLM API调用失败 (尝试了 {attempt + 1} 次): {error_msg}") from last_exception
        
        # 不应该到达这里，但为了安全起见
        raise Exception(f"LLM API调用失败: {str(last_exception)}") from last_exception
    
    def _extract_response_content(self, response) -> str:
        """从响应中提取内容"""
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            if content is None:
                logger.warning("Response content is None, returning empty string")
                return ""
            return content
        elif isinstance(response, str):
            return response
        elif isinstance(response, dict):
            if 'choices' in response and response['choices']:
                content = response['choices'][0]['message']['content']
                if content is None:
                    logger.warning("Response content is None, returning empty string")
                    return ""
                return content
            elif 'content' in response:
                content = response['content']
                if content is None:
                    logger.warning("Response content is None, returning empty string")
                    return ""
                return content
            elif 'message' in response:
                message = response['message']
                if message is None:
                    logger.warning("Response message is None, returning empty string")
                    return ""
                return message
            else:
                return str(response)
        else:
            return str(response)

    def _log_debug_request(self, request_messages):
        print("Request Messages Begin")
        for m in request_messages:
            print("---------------------")
            print(m)

        print("Request Messages End")
            
    def _log_debug_response(self, response, content: str, attempt: int):
        """记录详细的响应调试信息"""
        attempt_info = f" (尝试 {attempt + 1})" if attempt > 0 else ""
        print(response)

    
    def _should_retry_on_error(self, error: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        error_str = str(error).lower()
        
        # 网络相关错误 - 应该重试
        network_errors = [
            'timeout', 'connection', 'network', 'refused', 'unreachable',
            'temporary failure', 'service unavailable', '502', '503', '504'
        ]
        
        # API限制相关错误 - 应该重试
        rate_limit_errors = [
            'rate limit', 'too many requests', '429', 'quota exceeded'
        ]
        
        # 服务器错误 - 应该重试  
        server_errors = [
            'internal server error', '500', '502', '503', '504',
            'bad gateway', 'gateway timeout'
        ]
        
        # 客户端错误 - 不应该重试
        client_errors = [
            'invalid api key', 'unauthorized', '401', '403', '404',
            'bad request', '400', 'invalid parameter'
        ]
        
        # 检查是否为不应重试的错误
        for client_error in client_errors:
            if client_error in error_str:
                logger.debug(f"   🚫 客户端错误，不重试: {client_error}")
                return False
        
        # 检查是否为应该重试的错误
        all_retryable_errors = network_errors + rate_limit_errors + server_errors
        for retryable_error in all_retryable_errors:
            if retryable_error in error_str:
                logger.debug(f"   🔄 可重试错误: {retryable_error}")
                return True
        
        # 默认情况：如果是未知错误且还有重试次数，则重试
        logger.debug(f"   ❓ 未知错误类型，默认重试")
        return True
    
    async def close(self):
        """关闭客户端连接"""
        try:
            if hasattr(self.client, 'aclose'):
                await self.client.aclose()
            elif hasattr(self.client, 'close'):
                await self.client.close()
        except Exception as e:
            logger.warning(f"Error closing client: {e}")


def create_llm_client(api_key: str, base_url: str, model: str, debug: bool = False, max_retries: int = 3,
                     proxy_url: Optional[str] = None) -> LLMAPIClient:
    """
    Factory function to create LLM API client.
    
    Args:
        api_key: API key for authentication
        base_url: Base URL of the API endpoint  
        model: Model name to use for generation
        debug: Enable debug mode to log all API inputs and outputs
        max_retries: Maximum number of retry attempts on failure (default: 3)
        proxy_url: Optional proxy URL to use for requests (overrides environment variables)
        
    Returns:
        Configured LLMAPIClient instance
    """
    return LLMAPIClient(api_key, base_url, model, debug, max_retries, proxy_url)


async def test_llm_connection(client: LLMAPIClient) -> bool:
    """
    Test LLM API connection.
    
    Args:
        client: LLMAPIClient instance to test
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        test_messages = [
            {"role": "user", "content": "Hello, can you respond with 'API connection successful'?"}
        ]
        
        response = await client.generate(test_messages, max_tokens=50)
        logger.info(f"API测试响应: {response}")
        
        if "successful" in response.lower() or "connection" in response.lower():
            logger.info("✅ API连接测试成功!")
            return True
        else:
            logger.warning("⚠️ API连接可能有问题，但收到了响应")
            return True
            
    except Exception as e:
        logger.error(f"❌ API连接测试失败: {e}")
        return False

