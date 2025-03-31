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

# 构建工作流图，使用更强大的模型获取更详细的答案
rag.build_graph(model_name="gpt-4o-mini")

# 定义多个具体的查询问题
queries = [
    "可塑性机制有哪些不同类型？请详细说明每种类型的特点。",
    "突触可塑性和内在可塑性有什么区别？它们如何相互作用？",
    "可塑性机制在回声状态网络(Echo State Networks)中扮演什么角色？",
    "可塑性机制如何影响神经网络的学习能力和记忆形成？"
]

# 执行每个查询
for i, query in enumerate(queries):
    print(f"\n\n===== 查询 {i+1}: {query} =====\n")
    
    # 直接获取最终答案
    answer = rag.run(query)
    print(f"回答:\n{answer}")
    
    print("\n" + "-"*80)

print("\n===== 查询完成 =====") 