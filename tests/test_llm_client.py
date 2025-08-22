#!/usr/bin/env python3
"""
测试LLMAPIClient的功能
演示如何使用通用LLM客户端
"""

import asyncio
import sys
import os

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.utils import create_llm_client


async def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试LLMAPIClient基本功能 ===")
    
    # API配置
    api_key = "**************************************"
    base_url = "**********************"
    model = "claude-sonnet-4-20250514"
    
    # 1. 使用工厂函数创建客户端
    print("\n1. 使用create_llm_client工厂函数...")
    client1 = create_llm_client(
        api_key=api_key,
        base_url=base_url,
        model=model,
        debug=False,
        max_retries=2
    )
    
    # 2. 创建第二个客户端实例
    print("2. 创建第二个客户端实例...")
    client2 = create_llm_client(
        api_key=api_key,
        base_url=base_url,
        model=model,
        debug=True,
        max_retries=1
    )
    
    try:
        # 3. 测试连接
        print("\n3. 测试API连接...")
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            response = await client1.generate(test_messages, max_tokens=10)
            print(f"连接状态: ✅ 成功 (响应: {response[:30]}...)")
        except Exception as e:
            print(f"连接状态: ❌ 失败 ({e})")
        
        # 4. 测试简单对话
        print("\n4. 测试简单对话...")
        messages = [
            {"role": "user", "content": "请简单介绍一下什么是ReAct框架"}
        ]
        
        response = await client1.generate(messages, max_tokens=200, temperature=0.3)
        print(f"响应: {response[:200]}...")
        
        # 5. 测试多轮对话
        print("\n5. 测试多轮对话...")
        conversation = [
            {"role": "user", "content": "你好，我想了解人工智能"},
            {"role": "assistant", "content": "你好！我很乐意为您介绍人工智能。"},
            {"role": "user", "content": "请简单解释一下机器学习"}
        ]
        
        response = await client2.generate(conversation, max_tokens=150, temperature=0.5)
        print(f"多轮对话响应: {response[:150]}...")
        
    finally:
        # 清理资源
        await client1.close()
        await client2.close()


async def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试错误的API配置
    print("\n1. 测试错误的API配置...")
    try:
        bad_client = create_llm_client(
            api_key="invalid-key",
            base_url="http://invalid-url",
            model="invalid-model",
            max_retries=1
        )
        response = await bad_client.generate([{"role": "user", "content": "test"}], max_tokens=10)
        print(f"错误处理响应: {response[:100]}...")
        await bad_client.close()
    except Exception as e:
        print(f"捕获到异常: {e}")


async def test_different_parameters():
    """测试不同参数配置"""
    print("\n=== 测试不同参数配置 ===")
    
    api_key = "sk-qq7xJtnAdB1Gv6IkHTQhDAPuUAT700vF3CMmGinILsmP2HuY"
    base_url = "http://211.23.3.237:27544"
    model = "claude-sonnet-4-20250514"
    
    client = create_llm_client(api_key, base_url, model)
    
    try:
        # 1. 测试不同温度
        print("\n1. 测试不同温度参数...")
        messages = [{"role": "user", "content": "用一句话描述春天"}]
        
        for temp in [0.1, 0.7, 1.0]:
            response = await client.generate(messages, max_tokens=50, temperature=temp)
            print(f"温度 {temp}: {response[:80]}...")
        
        # 2. 测试不同token限制
        print("\n2. 测试不同token限制...")
        messages = [{"role": "user", "content": "请详细解释什么是人工智能"}]
        
        for tokens in [50, 100, 200]:
            response = await client.generate(messages, max_tokens=tokens, temperature=0.5)
            print(f"最大tokens {tokens}: {response[:100]}...")
            
    finally:
        await client.close()


async def main():
    """主函数"""
    print("LLMAPIClient 功能测试")
    print("=" * 50)
    
    try:
        await test_basic_functionality()
        await test_error_handling() 
        await test_different_parameters()
        
        print("\n✅ 所有测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
    🧪 LLM客户端测试说明:
    
    本测试将验证LLMAPIClient的以下功能：
    1. 基本实例化和工厂函数创建
    2. API连接测试
    3. 简单对话生成
    4. 多轮对话支持
    5. 错误处理机制
    6. 不同参数配置
    
    测试使用的API端点:
    - Base URL: http://211.23.3.237:27544/
    - Model: claude-sonnet-4-20250514
    """)
    
    asyncio.run(main())
