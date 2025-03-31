import os
from agentic_rag import AgenticRAG

# 设置OpenAI API密钥
# 注意：请确保已设置环境变量OPENAI_API_KEY或在此处提供
openai_api_key = os.environ.get("OPENAI_API_KEY")

# 创建AgenticRAG实例
rag = AgenticRAG(openai_api_key)

# 使用doc文件夹中的PDF文件创建检索器
pdf_directory = "./doc"
rag.create_retriever(pdf_directory=pdf_directory)

# 构建工作流图
rag.build_graph(model_name="gpt-4o-mini")  # 可以根据需要更改模型

# 执行查询：可塑性机制是什么
query = "可塑性机制是什么？它在神经网络中扮演什么角色？"
print(f"\n查询: {query}")

# 流式运行查询，展示中间步骤
print("\n处理查询过程:")
for step in rag.stream_run(query):
    node = step["node"]
    if node == "generate":
        print(f"\n最终回答:")
        if hasattr(step["output"]["messages"][0], 'content'):
            print(step["output"]["messages"][0].content)
        else:
            print(step["output"]["messages"][0])
    else:
        print(f"节点: {node}")

print("\n=== 完成 ===") 