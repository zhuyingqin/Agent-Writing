import os
import sys
import math
import contextlib
import io
import builtins
import getpass
import time
import signal
import traceback
from typing import Any, Callable, Dict, List, Tuple, Union, Optional, Set

# 数学工具函数
def add(a: float, b: float) -> float:
    """将两个数字相加。"""
    return a + b

# 默认工具列表
DEFAULT_TOOLS = [
    add,
]

# 代码沙箱函数
def create_sandbox_eval() -> Callable[[str, Dict[str, Any]], Tuple[str, Dict[str, Any]]]:
    """
    创建一个代码沙箱评估函数。
    
    在生产环境中应使用安全的沙箱环境！此处的eval函数仅用于演示目的，不安全！
    
    返回:
        代码沙箱评估函数
    """
    # 同步实现
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

def create_codeact_agent(
    model, 
    tools: Optional[List] = None, 
    sandbox_eval: Optional[Callable] = None, 
    prompt: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    handle_serialization_errors: bool = True
):
    """
    创建一个CodeAct代理。
    
    参数:
        model: LangChain兼容的语言模型
        tools: 工具函数列表（如果为None，则使用默认工具）
        sandbox_eval: 代码沙箱评估函数（如果为None，则创建一个新的）
        prompt: 自定义提示（如果需要）
        max_retries: 操作失败时的最大重试次数
        retry_delay: 重试间隔（秒）
        handle_serialization_errors: 是否自动处理序列化错误
        
    返回:
        编译后的CodeAct代理，增强了重试和错误处理功能
    """
    from langgraph_codeact import create_codeact
    from langgraph.checkpoint.memory import MemorySaver
    import time
    import pprint
    
    if tools is None:
        tools = DEFAULT_TOOLS
    
    if sandbox_eval is None:
        sandbox_eval = create_sandbox_eval()
    
    # 创建基础代理
    code_act = create_codeact(model, tools, sandbox_eval, prompt=prompt)
    base_agent = code_act.compile(checkpointer=MemorySaver())
    
    # 创建具有重试能力的包装代理
    class RetryableAgent:
        def __init__(self, agent, max_retries=3, retry_delay=2.0, handle_serialization_errors=True):
            self.agent = agent
            self.max_retries = max_retries
            self.retry_delay = retry_delay
            self.handle_serialization_errors = handle_serialization_errors
        
        def invoke(self, inputs, config=None):
            current_retry_delay = self.retry_delay
            
            for attempt in range(1, self.max_retries + 1):
                try:
                    if attempt > 1:
                        print(f"\n尝试 {attempt}/{self.max_retries}...")
                    
                    result = self.agent.invoke(inputs, config=config)
                    return result
                    
                except TypeError as e:
                    if self.handle_serialization_errors and "is not serializable" in str(e):
                        print(f"序列化错误: {e}")
                        print("这可能是由于代码生成或执行过程中创建了不可序列化的对象")
                        
                        # 如果是最后一次尝试，向上抛出错误
                        if attempt == self.max_retries:
                            print("已达到最大重试次数，操作失败。")
                            raise
                            
                        print(f"将在 {current_retry_delay:.1f} 秒后重试...")
                        time.sleep(current_retry_delay)
                        # 指数退避策略
                        current_retry_delay *= 1.5
                    else:
                        # 非序列化错误直接抛出
                        raise
                        
                except Exception as e:
                    print(f"错误: {type(e).__name__}: {str(e)}")
                    
                    # 如果是最后一次尝试，向上抛出错误
                    if attempt == self.max_retries:
                        print("已达到最大重试次数，操作失败。")
                        raise
                        
                    print(f"将在 {current_retry_delay:.1f} 秒后重试...")
                    time.sleep(current_retry_delay)
                    # 指数退避策略
                    current_retry_delay *= 1.5
            
            # 这里应该永远不会到达，因为最后一次尝试失败会抛出异常
            raise RuntimeError("重试机制逻辑错误")
        
        def stream(self, inputs, stream_mode=None, config=None):
            current_retry_delay = self.retry_delay
            
            for attempt in range(1, self.max_retries + 1):
                try:
                    if attempt > 1:
                        print(f"\n尝试流式处理 {attempt}/{self.max_retries}...")
                    
                    # 创建一个带有错误诊断的流式生成器
                    stream_gen = self.agent.stream(inputs, stream_mode=stream_mode, config=config)
                    
                    # 处理流式结果
                    for typ, chunk in stream_gen:
                        # 如果启用了序列化错误处理，尝试检测和修复问题
                        if self.handle_serialization_errors and typ == "values" and "context" in chunk and isinstance(chunk["context"], dict):
                            # 诊断序列化问题
                            all_issues = []
                            for key, value in chunk["context"].items():
                                issues = diagnose_serialization_issues(value, f"context.{key}")
                                all_issues.extend(issues)
                            
                            # 如果发现问题，输出诊断信息
                            if all_issues:
                                print("\n检测到流式输出中的序列化问题:")
                                pprint.pprint(all_issues)
                            
                            # 过滤不可序列化的对象
                            filtered_chunk = {}
                            for k, v in chunk.items():
                                if k == "context" and isinstance(v, dict):
                                    filtered_chunk[k] = filter_non_serializable(v)
                                else:
                                    filtered_chunk[k] = v
                            
                            # 返回过滤后的数据
                            yield typ, filtered_chunk
                        else:
                            # 直接传递其他类型的数据
                            yield typ, chunk
                    
                    # 如果流式处理成功完成，退出重试循环
                    break
                    
                except TypeError as e:
                    if self.handle_serialization_errors and "is not serializable" in str(e):
                        print(f"流式处理序列化错误: {e}")
                        
                        # 如果是最后一次尝试，向上抛出错误
                        if attempt == self.max_retries:
                            print("已达到最大重试次数，流式处理失败。")
                            raise
                            
                        print(f"将在 {current_retry_delay:.1f} 秒后重试流式处理...")
                        time.sleep(current_retry_delay)
                        # 指数退避策略
                        current_retry_delay *= 1.5
                    else:
                        # 非序列化错误直接抛出
                        raise
                        
                except Exception as e:
                    print(f"流式处理错误: {type(e).__name__}: {str(e)}")
                    
                    # 如果是最后一次尝试，向上抛出错误
                    if attempt == self.max_retries:
                        print("已达到最大重试次数，流式处理失败。")
                        raise
                        
                    print(f"将在 {current_retry_delay:.1f} 秒后重试流式处理...")
                    time.sleep(current_retry_delay)
                    # 指数退避策略
                    current_retry_delay *= 1.5
    
    # 返回增强的代理
    return RetryableAgent(base_agent, max_retries=max_retries, retry_delay=retry_delay, 
                         handle_serialization_errors=handle_serialization_errors)

# 添加检查非序列化对象的函数
def filter_non_serializable(obj_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    过滤掉不可序列化的对象，比如模块类型、文件句柄等
    
    参数:
        obj_dict: 包含各种对象的字典
        
    返回:
        只包含可序列化对象的字典
    """
    import io
    import types
    
    result = {}
    for key, value in obj_dict.items():
        # 过滤掉模块类型
        if isinstance(value, type(math)) or isinstance(value, types.ModuleType):
            continue
            
        # 过滤掉文件类型
        if isinstance(value, io.IOBase) or str(type(value)) == "<class '_io.TextIOWrapper'>" or hasattr(value, 'read'):
            continue
            
        # 过滤掉有__module__属性并且模块是builtins的对象
        if hasattr(value, "__module__") and value.__module__ == "builtins" and callable(value):
            continue
        
        # 过滤迭代器和生成器
        if isinstance(value, types.GeneratorType) or hasattr(value, '__iter__') and callable(value.__iter__) and not isinstance(value, (list, dict, set, tuple, str)):
            continue
        
        # 递归处理字典
        if isinstance(value, dict):
            result[key] = filter_non_serializable(value)
            continue
        
        # 递归处理列表和元组
        if isinstance(value, (list, tuple)):
            filtered_list = []
            all_serializable = True
            for item in value:
                if isinstance(item, (dict, list, tuple, set)):
                    if isinstance(item, dict):
                        filtered_list.append(filter_non_serializable(item))
                    else:
                        # 这里可以进一步递归处理，但为简单起见，我们只添加可序列化的基本类型
                        try:
                            import msgpack
                            msgpack.packb(item, default=lambda x: None)
                            filtered_list.append(item)
                        except (TypeError, ValueError):
                            all_serializable = False
                            break
                else:
                    try:
                        import msgpack
                        msgpack.packb(item, default=lambda x: None)
                        filtered_list.append(item)
                    except (TypeError, ValueError):
                        # 跳过不可序列化的项
                        continue
            
            if all_serializable:
                if isinstance(value, tuple):
                    result[key] = tuple(filtered_list)
                else:
                    result[key] = filtered_list
            continue
            
        try:
            # 简单测试是否可序列化
            import msgpack
            msgpack.packb(value, default=lambda x: None)
            result[key] = value
        except (TypeError, ValueError):
            # 如果不可序列化，则跳过
            pass
    return result

def diagnose_serialization_issues(obj, path="root"):
    """
    深度分析对象中的序列化问题
    
    参数:
        obj: 要分析的对象
        path: 当前对象的路径
        
    返回:
        问题列表，每个问题包含路径和问题描述
    """
    import io
    import types
    import inspect
    
    issues = []
    
    # 检查基本类型
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return issues
    
    # 检查文件对象
    if isinstance(obj, io.IOBase) or str(type(obj)) == "<class '_io.TextIOWrapper'>" or hasattr(obj, 'read') and callable(obj.read):
        issues.append({
            "path": path,
            "type": type(obj).__name__,
            "issue": "文件对象不可序列化",
            "solution": "关闭文件并移除引用，或使用文件名代替"
        })
        return issues
    
    # 检查模块
    if isinstance(obj, types.ModuleType):
        issues.append({
            "path": path,
            "type": type(obj).__name__,
            "issue": "模块对象不可序列化",
            "solution": "移除模块引用，只保留需要的函数或值"
        })
        return issues
    
    # 检查函数和方法
    if inspect.isfunction(obj) or inspect.ismethod(obj) or callable(obj):
        issues.append({
            "path": path,
            "type": type(obj).__name__,
            "issue": "函数或方法不可序列化",
            "solution": "移除函数引用，或将函数转换为字符串表示"
        })
        return issues
    
    # 检查生成器和迭代器
    if isinstance(obj, types.GeneratorType) or (hasattr(obj, '__iter__') and callable(obj.__iter__) and not isinstance(obj, (list, dict, set, tuple, str))):
        issues.append({
            "path": path,
            "type": type(obj).__name__,
            "issue": "生成器或迭代器不可序列化",
            "solution": "将生成器或迭代器转换为列表"
        })
        return issues
    
    # 递归检查字典
    if isinstance(obj, dict):
        for key, value in obj.items():
            sub_path = f"{path}.{key}" if path != "root" else key
            issues.extend(diagnose_serialization_issues(value, sub_path))
        return issues
    
    # 递归检查列表或元组
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            sub_path = f"{path}[{i}]"
            issues.extend(diagnose_serialization_issues(item, sub_path))
        return issues
    
    # 检查自定义对象
    if hasattr(obj, "__dict__"):
        for attr_name, attr_value in obj.__dict__.items():
            if not attr_name.startswith("__"):  # 跳过内置属性
                sub_path = f"{path}.{attr_name}"
                issues.extend(diagnose_serialization_issues(attr_value, sub_path))
    
    # 对于其他类型，尝试序列化测试
    try:
        import msgpack
        msgpack.packb(obj, default=lambda x: None)
    except (TypeError, ValueError):
        issues.append({
            "path": path,
            "type": type(obj).__name__,
            "issue": "对象不可序列化",
            "solution": "转换为基本类型或移除此对象"
        })
    
    return issues

# 提供具有超时功能的同步沙箱
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
    import signal
    
    # 记录执行前的变量键
    original_keys = set(_locals.keys())
    _locals_copy = _locals.copy()  # 创建一个副本，避免原始字典被修改
    
    # 超时处理函数
    def timeout_handler(signum, frame):
        raise TimeoutError(f"代码执行超过{timeout}秒")
    
    # 设置信号处理器
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))
    
    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            # 执行代码
            exec(code, builtins.__dict__, _locals_copy)
        result = f.getvalue()
        if not result:
            result = "<代码已执行，但没有输出到stdout>"
        
        # 将本地变量的更改合并回原始字典
        _locals.update(_locals_copy)
        
        # 确定执行过程中创建的新变量
        new_keys = set(_locals.keys()) - original_keys
        new_vars = {key: _locals[key] for key in new_keys}
        
        # 过滤掉不可序列化的对象
        new_vars = filter_non_serializable(new_vars)
        
        return result, new_vars
        
    except TimeoutError as e:
        return f"执行超时: {str(e)}", {}
    except Exception as e:
        return f"执行错误: {repr(e)}", {}
    finally:
        # 取消alarm
        signal.alarm(0)

# 示例用法
def run_example():
    """运行一个简单的示例来演示CodeAct的使用。"""
    from langchain.chat_models import init_chat_model
    
    # 检查是否设置了OPENAI_API_KEY环境变量
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("请输入OPENAI_API_KEY: ")
        print("API密钥已设置")
    
    print("初始化模型...")
    model = init_chat_model("openai:gpt-4o-mini")
    
    # 创建代理时指定更安全的沙箱评估函数
    def safe_sandbox_eval(code, local_vars):
        # 尝试修复代码中的常见问题
        fixed_code, fixes = fix_common_code_issues(code)
        
        if fixes:
            print("\n修复了以下问题:")
            for fix in fixes:
                print(f"- {fix}")
            print("\n修复后的代码:")
            print(fixed_code)
            print()
            code = fixed_code
        
        # 使用我们的沙箱评估函数执行代码
        sandbox = create_sandbox_eval()
        
        try:
            result, new_vars = sandbox(code, local_vars)
            
            # 检查新变量是否有序列化问题
            for key, value in new_vars.items():
                issues = diagnose_serialization_issues(value, key)
                if issues:
                    print(f"\n变量 '{key}' 存在序列化问题:")
                    for issue in issues:
                        print(f"- 在 {issue['path']}: {issue['issue']} (类型: {issue['type']})")
                        print(f"  解决方案: {issue['solution']}")
            
            # 确保返回的变量是可序列化的
            filtered_vars = filter_non_serializable(new_vars)
            
            # 检查过滤前后的差异
            filtered_keys = set(filtered_vars.keys())
            original_keys = set(new_vars.keys())
            removed_keys = original_keys - filtered_keys
            
            if removed_keys:
                print(f"\n由于序列化问题，以下变量已被过滤:")
                for key in removed_keys:
                    print(f"- {key} (类型: {type(new_vars[key]).__name__})")
            
            return result, filtered_vars
            
        except Exception as e:
            print(f"代码执行错误: {type(e).__name__}: {str(e)}")
            # 返回错误信息和空字典
            return f"执行错误: {type(e).__name__}: {str(e)}", {}
    
    # 创建代理，启用重试机制和序列化错误处理
    print("创建代理...")
    agent = create_codeact_agent(
        model, 
        sandbox_eval=safe_sandbox_eval,
        max_retries=3,
        retry_delay=2.0,
        handle_serialization_errors=True
    )
    
    messages = [{
        "role": "user",
        "content": "构建一个"
    }]
    
    # 执行代理调用（现在已经内置了重试机制）
    print("执行模型调用...")
    result = agent.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": 1}}
    )
    
    print("结果:", result)
    
    # 流式输出示例（同样内置了重试机制）
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
    
    print("\n执行完成。")

def fix_common_code_issues(code: str) -> Tuple[str, List[str]]:
    """
    修复代码中的常见问题
    
    参数:
        code: 要修复的代码
        
    返回:
        修复后的代码和修复信息列表
    """
    code_lines = code.split('\n')
    fixes = []
    
    # 修复文件操作问题
    fixed_file_handling = fix_file_operations(code_lines)
    if fixed_file_handling[1]:
        code_lines = fixed_file_handling[0]
        fixes.extend(fixed_file_handling[1])
    
    # 修复资源释放问题
    fixed_resources = fix_resource_cleanup(code_lines)
    if fixed_resources[1]:
        code_lines = fixed_resources[0]
        fixes.extend(fixed_resources[1])
    
    # 修复异常处理问题
    fixed_exceptions = fix_exception_handling(code_lines)
    if fixed_exceptions[1]:
        code_lines = fixed_exceptions[0]
        fixes.extend(fixed_exceptions[1])
    
    return '\n'.join(code_lines), fixes

def fix_file_operations(code_lines: List[str]) -> Tuple[List[str], List[str]]:
    """修复文件操作问题"""
    fixes = []
    file_vars = {}  # 跟踪需要关闭的文件变量
    fixed_lines = code_lines.copy()
    
    # 第一遍：识别文件打开操作
    for i, line in enumerate(code_lines):
        # 检测文件打开操作
        if 'open(' in line and 'with' not in line:
            indent = len(line) - len(line.lstrip())
            leading_space = " " * indent
            
            # 提取文件操作信息
            if '=' in line:
                var_name = line.split('=')[0].strip()
                file_op = line.split('=')[1].strip()
                
                # 记录要关闭的文件
                file_vars[var_name] = i
                
                # 尝试构建with语句替换
                try:
                    with_line = f"{leading_space}with {file_op} as {var_name}:"
                    fixed_lines[i] = with_line
                    
                    # 查找此文件变量的所有使用点，确保它们被包含在with块中
                    current_block_end = len(code_lines)
                    for j in range(i+1, len(code_lines)):
                        next_line = code_lines[j]
                        next_indent = len(next_line) - len(next_line.lstrip())
                        
                        # 如果缩进级别降低，则找到了块的结束
                        if next_indent <= indent and next_line.strip():
                            current_block_end = j
                            break
                    
                    # 确保后续的代码有正确的缩进
                    for j in range(i+1, current_block_end):
                        if code_lines[j].strip():  # 不处理空行
                            next_line = code_lines[j]
                            next_indent = len(next_line) - len(next_line.lstrip())
                            
                            if next_indent <= indent:  # 如果缩进不足
                                fixed_lines[j] = leading_space + "    " + next_line.lstrip()
                    
                    fixes.append(f"第{i+1}行: 文件操作 '{var_name}' 已转换为with语句")
                except Exception:
                    # 如果自动修复失败，记录警告
                    fixes.append(f"第{i+1}行: 文件操作 '{var_name}' 需要关闭但无法自动修复")
    
    # 第二遍：检查是否有未关闭的文件
    for var_name, line_num in file_vars.items():
        # 查找close操作
        close_found = False
        for line in code_lines:
            if f"{var_name}.close()" in line:
                close_found = True
                break
        
        if not close_found:
            # 找到变量最后一次使用的位置
            last_use = line_num
            for i, line in enumerate(code_lines):
                if var_name in line and i > last_use:
                    last_use = i
            
            # 添加close语句
            indent = len(code_lines[last_use]) - len(code_lines[last_use].lstrip())
            leading_space = " " * indent
            close_line = f"{leading_space}{var_name}.close()"
            
            # 如果我们没有将其转换为with语句，则添加close
            if "with" not in fixed_lines[line_num]:
                fixed_lines.insert(last_use + 1, close_line)
                fixes.append(f"在第{last_use+2}行: 添加 '{var_name}.close()' 语句")
    
    return fixed_lines, fixes

def fix_resource_cleanup(code_lines: List[str]) -> Tuple[List[str], List[str]]:
    """修复资源清理问题"""
    fixes = []
    # 此处可以扩展更多资源清理逻辑
    return code_lines, fixes

def fix_exception_handling(code_lines: List[str]) -> Tuple[List[str], List[str]]:
    """修复异常处理问题"""
    fixes = []
    
    # 查找try块但无except
    in_try_block = False
    try_start = -1
    try_indent = 0
    has_except = False
    
    for i, line in enumerate(code_lines):
        stripped = line.strip()
        
        if stripped.startswith("try:"):
            in_try_block = True
            try_start = i
            try_indent = len(line) - len(line.lstrip())
        elif in_try_block and stripped.startswith("except"):
            has_except = True
        elif in_try_block and line.strip() and len(line) - len(line.lstrip()) <= try_indent:
            # 离开try块的缩进级别
            if not has_except:
                # 添加基本的异常处理
                leading_space = " " * try_indent
                new_except = [
                    f"{leading_space}except Exception as e:",
                    f"{leading_space}    print(f\"错误: {{e}}\")"
                ]
                code_lines = code_lines[:i] + new_except + code_lines[i:]
                fixes.append(f"在第{i}行添加异常处理")
                # 调整索引
                i += len(new_except)
            
            in_try_block = False
            try_start = -1
            has_except = False
    
    return code_lines, fixes

if __name__ == "__main__":
    print("导入此模块以使用CodeAct功能，或直接运行此文件以查看示例。")
    
    # 测试SVG代码解析能力
    # test_svg_code()
    
    # 如果需要API测试，取消注释以下行
    run_example()  # 需要OpenAI API密钥

