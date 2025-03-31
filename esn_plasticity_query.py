import os
from agentic_rag import AgenticRAG

# 设置OpenAI API密钥
openai_api_key = os.environ.get("OPENAI_API_KEY")

# 创建AgenticRAG实例
rag = AgenticRAG(openai_api_key)

# 使用doc文件夹中的PDF文件创建检索器
pdf_directory = "./doc"
rag.create_retriever(pdf_directory=pdf_directory)

# 构建工作流图
rag.build_graph(model_name="gpt-4o-mini")

# 更具体的关于回声状态网络中可塑性机制的查询
esn_queries = [
    "回声状态网络(Echo State Networks)中如何实现可塑性机制？",
    "在回声状态网络中，突触可塑性和内在可塑性如何协同工作？",
    "回声状态网络中的可塑性机制与传统神经网络有什么不同？",
    "请详细解释回声状态网络中的突触可塑性和内在可塑性的实现方法"
]

# 执行每个查询
print("\n\n===== 回声状态网络(ESN)中的可塑性机制研究 =====\n")

for i, query in enumerate(esn_queries):
    print(f"\n----- 查询 {i+1}: {query} -----\n")
    
    # 直接获取最终答案
    answer = rag.run(query)
    print(f"回答:\n{answer}")
    
    print("\n" + "-"*80)

# 尝试更具体的查询，针对文件名中含有Echo State Networks的PDF
print("\n\n===== 分析文件 'Synergies between synaptic and intrinsic plasticity in echo state networks.pdf' =====\n")

specific_query = "根据文件'Synergies between synaptic and intrinsic plasticity in echo state networks.pdf'，详细说明回声状态网络中突触可塑性和内在可塑性的协同作用"
answer = rag.run(specific_query)
print(f"回答:\n{answer}")

print("\n===== 查询完成 =====") 