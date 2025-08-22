#!/usr/bin/env python3
"""
简单的 LLM 调用函数 - 使用 requests 库
"""

import os
import requests
from typing import List, Dict, Optional
import logging
import time
import urllib.parse


def call_llm(messages: List[Dict[str, str]], 
             model: str = None,
             temperature: float = 1.0,
             top_p: float = 0.7,
             max_tokens: Optional[int] = None,
             timeout: int = 60,
             max_retries: int = 5) -> str:
    """
    调用 LLM API，支持重试机制
    
    Args:
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
        model: 模型名称，默认从环境变量获取
        temperature: 采样温度
        top_p: 核采样参数
        max_tokens: 最大生成 token 数
        timeout: 超时时间（秒）
        max_retries: 最大重试次数，默认5次
        
    Returns:
        生成的文本内容
    """
    # 获取配置
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = model or os.getenv("LLM_MODEL_NAME")
    
    # 验证必需的环境变量
    if not api_key:
        raise ValueError("LLM_API_KEY environment variable is required")
    if not base_url:
        raise ValueError("LLM_BASE_URL environment variable is required")
    if not model:
        raise ValueError("LLM_MODEL_NAME environment variable is required or model parameter must be provided")
    
    # URL - 检查 base_url 是否已经包含路径

    
    if base_url.endswith('/v1/chat/completions') or base_url.endswith('/v2/chat/completions'):
        url = base_url
    elif base_url.endswith('/v1'):
        url = f"{base_url}/chat/completions"
    
    # 请求头
    headers = {
        'Authorization': "Bearer " + api_key,
        'Content-Type': 'application/json'
    }
    
    # 请求体
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    
    if max_tokens:
        data["max_tokens"] = max_tokens
    
    # 检查是否是内网地址（不需要代理）
    
    # 重试逻辑
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # 只在第一次尝试时打印详细信息
            if attempt == 1:
                logging.info(f"LLM endpoint: base_url={base_url}, final_url={url}")
                logging.info(f"Request model: {model}, max_retries: {max_retries}")
            
            # 发送请求
            logging.info(f"url: {url}")
            logging.info(f"headers: {headers}")
            logging.info(f"data: {data}")

            response = requests.post(
                url,
                headers=headers,
                json=data
            )
            
            logging.info("response content")
            logging.info(response.json())
            
            # 检查响应
            if response.status_code == 200:
                # 成功，返回结果
                if attempt > 1:
                    logging.info(f"LLM call succeeded on attempt {attempt}")
                return response.json()['choices'][0]['message']['content']
            else:
                # 非200状态码，记录错误但不打印（除非是最后一次）
                last_error = f"API returned {response.status_code}: {response.text}"
                
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            
        except Exception as e:
            last_error = f"Unexpected error: {e}"
        
        # 如果不是最后一次尝试，等待后重试
        if attempt < max_retries:
            # 指数退避：1秒, 2秒, 4秒, 8秒...
            wait_time = 60
            time.sleep(wait_time)
    
    # 所有尝试都失败了，打印错误并抛出异常
    logging.error(f"LLM call failed after {max_retries} attempts")
    logging.error(f"Last error: {last_error}")
    raise Exception(f"Failed to call LLM after {max_retries} attempts. Last error: {last_error}")


def ask_llm(question: str, **kwargs) -> str:
    """
    简单问答接口
    
    Args:
        question: 问题文本
        **kwargs: 其他参数传递给 call_llm（包括max_retries）
        
    Returns:
        回答文本
    """
    messages = [{"role": "user", "content": question}]
    return call_llm(messages, **kwargs)


# 测试代码
if __name__ == "__main__":
    print("测试 LLM 调用")
    print("=" * 60)
    
    # 测试 1: 基本调用
    try:
        response = ask_llm("给我写一个笑话")
        print(f"回答: {response}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试 2: 完整调用
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello World'"}
        ]
        response = call_llm(messages, max_tokens=20)
        print(f"回答: {response}")
    except Exception as e:
        print(f"错误: {e}")
