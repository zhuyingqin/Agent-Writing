import math
import asyncio
import contextlib
import io
import builtins
from typing import Any, Callable, Dict, List, Tuple, Union, Optional, Awaitable

# 数学工具函数
def add(a: float, b: float) -> float:
    """将两个数字相加。"""
    return a + b

# 默认工具列表
DEFAULT_TOOLS = [
    add,
]

# 代码沙箱函数
def create_sandbox_eval(async_support: bool = False) -> Union[
    Callable[[str, Dict[str, Any]], Tuple[str, Dict[str, Any]]],
    Callable[[str, Dict[str, Any]], Awaitable[Tuple[str, Dict[str, Any]]]]
]:
    """
    创建一个代码沙箱评估函数。
    
    在生产环境中应使用安全的沙箱环境！此处的eval函数仅用于演示目的，不安全！
    
    参数:
        async_support: 是否返回支持异步的评估函数
        
    返回:
        代码沙箱评估函数（同步或异步版本）
    """
    if not async_support:
        # 原始同步实现
        def eval_function(code: str, _locals: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            # 存储执行前的键
            original_keys = set(_locals.keys())

            try:
                with contextlib.redirect_stdout(io.StringIO()) as f:
                    exec(code, builtins.__dict__, _locals)
                result = f.getvalue()
                if not result:
                    result = "<代码已运行，但没有输出到stdout>"
            except Exception as e:
                result = f"执行错误: {repr(e)}"

            # 确定执行期间创建的新变量
            new_keys = set(_locals.keys()) - original_keys
            new_vars = {key: _locals[key] for key in new_keys}
            
            # 过滤掉不可序列化的对象
            new_vars = filter_non_serializable(new_vars)
            
            return result, new_vars
        
        return eval_function
    else:
        # 异步实现
        async def async_eval(code: str, _locals: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            """
            异步环境下执行代码，但使用与原始eval函数相同的执行方式
            
            参数:
                code: 要执行的代码字符串
                _locals: 本地变量字典
                
            返回:
                tuple: (执行结果输出, 新创建的变量字典)
            """
            # 记录执行前的变量键
            original_keys = set(_locals.keys())
            
            # 创建一个Future对象，用于在事件循环中执行同步代码
            loop = asyncio.get_running_loop()
            
            def execute_code():
                try:
                    with contextlib.redirect_stdout(io.StringIO()) as f:
                        exec(code, builtins.__dict__, _locals)
                    result = f.getvalue()
                    if not result:
                        result = "<代码已执行，但没有输出到stdout>"
                    return result, None
                except Exception as e:
                    return f"执行错误: {repr(e)}", e
            
            # 在事件循环的线程池中执行同步代码
            result, error = await loop.run_in_executor(None, execute_code)
            
            # 确定执行过程中创建的新变量
            new_keys = set(_locals.keys()) - original_keys
            new_vars = {key: _locals[key] for key in new_keys}
            
            # 过滤掉不可序列化的对象
            new_vars = filter_non_serializable(new_vars)
            
            return result, new_vars
            
        return async_eval

# 提供同步版本包装异步代码的函数
def sync_eval(code: str, _locals: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    同步版本的eval函数包装器，用于异步沙箱的同步调用
    
    参数:
        code: 要执行的代码字符串
        _locals: 本地变量字典
        
    返回:
        tuple: (执行结果输出, 新创建的变量字典)
    """
    # 获取或创建事件循环
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # 获取异步评估函数
    async_eval_func = create_sandbox_eval(async_support=True)
    
    # 运行异步函数并获取结果
    result, new_vars = loop.run_until_complete(async_eval_func(code, _locals))
    
    # 确保结果是可序列化的
    new_vars = filter_non_serializable(new_vars)
    
    return result, new_vars

def create_codeact_agent(
    model, 
    tools: Optional[List] = None, 
    sandbox_eval: Optional[Callable] = None, 
    async_sandbox: bool = False, 
    prompt: Optional[str] = None
):
    """
    创建一个CodeAct代理。
    
    参数:
        model: LangChain兼容的语言模型
        tools: 工具函数列表（如果为None，则使用默认工具）
        sandbox_eval: 代码沙箱评估函数（如果为None，则创建一个新的）
        async_sandbox: 是否使用异步沙箱（仅当sandbox_eval为None时有效）
        prompt: 自定义提示（如果需要）
        
    返回:
        编译后的CodeAct代理
    """
    from langgraph_codeact import create_codeact
    from langgraph.checkpoint.memory import MemorySaver
    
    if tools is None:
        tools = DEFAULT_TOOLS
    
    if sandbox_eval is None:
        sandbox_eval = create_sandbox_eval(async_support=async_sandbox)
    
    code_act = create_codeact(model, tools, sandbox_eval, prompt=prompt)
    agent = code_act.compile(checkpointer=MemorySaver())
    
    return agent

# 异步运行示例
async def run_async_example():
    """运行一个简单的异步示例来演示异步CodeAct的使用。"""
    from langchain.chat_models import init_chat_model
    
    model = init_chat_model("openai:gpt-4o")
    agent = create_codeact_agent(model, async_sandbox=True)
    
    messages = [{
        "role": "user",
        "content": "如果一个球以30米/秒的速度以45度角抛出，它会飞多远？"
    }]
    
    # 异步调用
    result = await agent.ainvoke(
        {"messages": messages},
        config={"configurable": {"thread_id": 3}}
    )
    
    print("异步结果:", result)
    
    # 异步流式输出示例
    print("\n异步流式输出示例:")
    async for typ, chunk in agent.astream(
        {"messages": messages},
        stream_mode=["values", "messages"],
        config={"configurable": {"thread_id": 4}},
    ):
        if typ == "messages":
            print(chunk[0].content, end="")
        elif typ == "values":
            print("\n\n---异步回答---\n\n", chunk)

# 示例用法
def run_example():
    """运行一个简单的示例来演示CodeAct的使用。"""
    from langchain.chat_models import init_chat_model
    
    model = init_chat_model("openai:gpt-4o")
    agent = create_codeact_agent(model)
    
    messages = [{
        "role": "user",
        "content": "如果一个球以30米/秒的速度以45度角抛出，它会飞多远？"
    }]
    
    result = agent.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": 1}}
    )
    
    print("结果:", result)
    
    # 流式输出示例
    print("\n流式输出示例:")
    for typ, chunk in agent.stream(
        {"messages": messages},
        stream_mode=["values", "messages"],
        config={"configurable": {"thread_id": 2}},
    ):
        if typ == "messages":
            print(chunk[0].content, end="")
        elif typ == "values":
            print("\n\n---回答---\n\n", chunk)

# 提供具有超时功能的异步沙箱
async def async_eval_with_timeout(
    code: str, 
    _locals: Dict[str, Any], 
    timeout: float = 5.0
) -> Tuple[str, Dict[str, Any]]:
    """
    带有超时功能的异步代码执行函数
    
    参数:
        code: 要执行的代码字符串
        _locals: 本地变量字典
        timeout: 超时时间（秒）
        
    返回:
        tuple: (执行结果输出, 新创建的变量字典)
    """
    from concurrent.futures import ThreadPoolExecutor
    
    # 记录执行前的变量键
    original_keys = set(_locals.keys())
    _locals_copy = _locals.copy()  # 创建一个副本，避免原始字典被修改
    
    # 创建一个Future对象，用于在事件循环中执行同步代码
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    
    # 为了支持超时，我们需要在单独的线程中执行代码
    def execute_code():
        try:
            with contextlib.redirect_stdout(io.StringIO()) as f:
                # 在执行代码前捕获当前时间
                exec(code, builtins.__dict__, _locals_copy)
            result = f.getvalue()
            if not result:
                result = "<代码已执行，但没有输出到stdout>"
            return result
        except Exception as e:
            return f"执行错误: {repr(e)}"
    
    try:
        # 使用asyncio.wait_for添加超时
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, execute_code),
            timeout=timeout
        )
        
        # 将本地变量的更改合并回原始字典
        _locals.update(_locals_copy)
        
        # 确定执行过程中创建的新变量
        new_keys = set(_locals.keys()) - original_keys
        new_vars = {key: _locals[key] for key in new_keys}
        
        # 过滤掉不可序列化的对象
        new_vars = filter_non_serializable(new_vars)
        
        return result, new_vars
        
    except asyncio.TimeoutError:
        executor.shutdown(wait=False)  # 立即关闭执行器，不等待任务完成
        return f"执行超时: 代码执行超过{timeout}秒", {}

# 同步版本的超时函数
def sync_eval_with_timeout(
    code: str, 
    _locals: Dict[str, Any], 
    timeout: float = 5.0
) -> Tuple[str, Dict[str, Any]]:
    """
    同步版本的带超时代码执行函数
    
    参数:
        code: 要执行的代码字符串
        _locals: 本地变量字典
        timeout: 超时时间（秒）
        
    返回:
        tuple: (执行结果输出, 新创建的变量字典)
    """
    # 获取或创建事件循环
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # 运行异步函数并获取结果
    return loop.run_until_complete(
        async_eval_with_timeout(code, _locals, timeout)
    )

# 添加检查非序列化对象的函数
def filter_non_serializable(obj_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    过滤掉不可序列化的对象，比如模块类型
    
    参数:
        obj_dict: 包含各种对象的字典
        
    返回:
        只包含可序列化对象的字典
    """
    result = {}
    for key, value in obj_dict.items():
        # 过滤掉模块类型
        if not isinstance(value, type(math)) and not (hasattr(value, "__module__") and value.__module__ == "builtins"):
            try:
                # 简单测试是否可序列化
                import msgpack
                msgpack.packb(value, default=lambda x: None)
                result[key] = value
            except (TypeError, ValueError):
                # 如果不可序列化，则跳过
                pass
    return result

    """
    测试沙箱功能，不依赖外部API
    """
    print("\n=== 沙箱功能测试 ===")
    
    # 创建同步沙箱
    sync_sandbox = create_sandbox_eval(async_support=False)
    
    # 简单的数学计算代码
    math_code = """
import math
result = 0
for i in range(10):
    result += i
result_sqrt = math.sqrt(result)
print(f"计算结果: 0-9的和 = {result}, 平方根 = {result_sqrt:.2f}")
"""
    
    # 使用沙箱执行代码
    print("\n1. 基本同步沙箱测试:")
    output, vars = sync_sandbox(math_code, {})
    print(output)
    print("新变量:", vars)
    
    # 测试完成
    print("\n=== 沙箱功能测试完成 ===")
    print("✓ 代码沙箱封装成功!")
    print("✓ 可以在没有API的情况下运行基本沙箱功能!")
    print("✓ 如需使用CodeAct代理，请配置API密钥并取消注释main函数中的run_example()!")

if __name__ == "__main__":
    print("导入此模块以使用CodeAct功能，或直接运行此文件以查看示例。")
    
    # 运行沙箱基本功能测试（不需要API）
    # test_sandbox_only()
    
    # 如果需要API测试，取消注释以下行
    run_example()  # 需要OpenAI API密钥

