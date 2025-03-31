import os
import logging
from typing import Annotated, Literal, Sequence, List, Optional, Union
from typing_extensions import TypedDict
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入必要的库
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

class AgenticRAG:
    """
    Agentic RAG系统类，用于集成所有组件
    """
    
    def __init__(self, openai_api_key=None):
        """
        初始化Agentic RAG系统
        
        参数:
            openai_api_key: OpenAI API密钥
        """
        # 设置API密钥
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError("请提供OpenAI API密钥")
        
        self.tools = []
        self.retriever = None
        self.graph = None
    
    def load_pdf_documents(self, pdf_paths: List[str]) -> List[Document]:
        """
        加载PDF文档
        
        参数:
            pdf_paths: PDF文件路径列表
            
        返回:
            文档列表
        """
        logger.info(f"正在加载{len(pdf_paths)}个PDF文档...")
        documents = []
        
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
                logger.info(f"已加载PDF: {pdf_path}")
            except Exception as e:
                logger.error(f"加载PDF {pdf_path} 时出错: {str(e)}")
        
        return documents
    
    def load_pdf_directory(self, directory_path: str) -> List[Document]:
        """
        加载目录中的所有PDF文档
        
        参数:
            directory_path: PDF文件目录
            
        返回:
            文档列表
        """
        logger.info(f"正在加载目录 {directory_path} 中的所有PDF文档...")
        try:
            loader = DirectoryLoader(
                directory_path, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            logger.info(f"已从目录加载 {len(documents)} 个文档")
            return documents
        except Exception as e:
            logger.error(f"加载目录 {directory_path} 中的PDF时出错: {str(e)}")
            return []
    
    def create_retriever(self, 
                        urls: Optional[List[str]] = None, 
                        pdf_paths: Optional[List[str]] = None,
                        pdf_directory: Optional[str] = None,
                        docs: Optional[List[Document]] = None, 
                        chunk_size: int = 500, 
                        chunk_overlap: int = 50):
        """
        创建检索器
        
        参数:
            urls: 网页URL列表
            pdf_paths: PDF文件路径列表
            pdf_directory: 包含PDF文件的目录
            docs: 已有的文档列表
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            
        返回:
            检索器
        """
        logger.info("正在创建检索器...")
        
        all_docs = []
        
        # 处理各种输入源
        if urls:
            # 从URL加载文档
            web_docs = [WebBaseLoader(url).load() for url in urls]
            web_docs_flat = [item for sublist in web_docs for item in sublist]
            logger.info(f"已加载 {len(urls)} 个网页URL，得到 {len(web_docs_flat)} 个文档")
            all_docs.extend(web_docs_flat)
        
        if pdf_paths:
            # 加载指定的PDF文件
            pdf_docs = self.load_pdf_documents(pdf_paths)
            logger.info(f"已加载 {len(pdf_docs)} 个PDF文档")
            all_docs.extend(pdf_docs)
            
        if pdf_directory:
            # 加载目录中所有PDF文件
            dir_docs = self.load_pdf_directory(pdf_directory)
            logger.info(f"已从目录加载 {len(dir_docs)} 个PDF文档")
            all_docs.extend(dir_docs)
        
        if docs:
            # 添加已有文档
            all_docs.extend(docs)
        
        if not all_docs:
            raise ValueError("请提供至少一种文档源：URLs、PDF文件、PDF目录或已有文档")
        
        logger.info(f"总共加载了 {len(all_docs)} 个文档")
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        doc_splits = text_splitter.split_documents(all_docs)
        logger.info(f"将文档分割为 {len(doc_splits)} 个文本块")
        
        # 创建向量数据库
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="agentic-rag-store",
            embedding=OpenAIEmbeddings(),
        )
        self.retriever = vectorstore.as_retriever()
        
        # 创建检索工具
        retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_documents",
            "搜索并返回文档中与查询相关的信息",
        )
        
        self.tools = [retriever_tool]
        logger.info("检索器创建完成")
        return self.retriever
    
    def build_graph(self, model_name="gpt-4o"):
        """
        构建工作流图
        
        参数:
            model_name: 使用的模型名称
        """
        logger.info("正在构建工作流图...")
        
        if not self.tools:
            raise ValueError("请先创建检索器")
        
        # 定义代理状态
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
        
        # 节点函数
        def agent(state):
            """调用代理模型生成响应或决定使用检索工具"""
            logger.info("调用代理")
            messages = state["messages"]
            model = ChatOpenAI(temperature=0, model=model_name)
            model = model.bind_tools(self.tools)
            response = model.invoke(messages)
            return {"messages": [response]}
        
        def grade_documents(state) -> Literal["generate", "rewrite"]:
            """评估检索文档是否与问题相关"""
            logger.info("检查文档相关性")
            
            class Grade(BaseModel):
                """相关性评分"""
                binary_score: str = Field(description="相关性评分 'yes' 或 'no'")
            
            model = ChatOpenAI(temperature=0, model=model_name)
            llm_with_tool = model.with_structured_output(Grade)
            
            prompt = PromptTemplate(
                template="""您是评估检索文档与用户问题相关性的评分员。\n 
                这是检索到的文档: \n\n {context} \n\n
                这是用户问题: {question} \n
                如果文档包含与用户问题相关的关键词或语义含义，将其评为相关。\n
                给出二元评分 'yes' 或 'no' 表示文档是否与问题相关。""",
                input_variables=["context", "question"],
            )
            
            chain = prompt | llm_with_tool
            
            messages = state["messages"]
            question = messages[0].content
            docs = messages[-1].content
            
            scored_result = chain.invoke({"question": question, "context": docs})
            score = scored_result.binary_score
            
            if score == "yes":
                logger.info("决定: 文档相关")
                return "generate"
            else:
                logger.info("决定: 文档不相关")
                return "rewrite"
        
        def rewrite(state):
            """改写查询以产生更好的问题"""
            logger.info("转换查询")
            messages = state["messages"]
            question = messages[0].content
            
            msg = [
                HumanMessage(
                    content=f"""\n 
            查看输入并尝试理解潜在的语义意图/含义。\n 
            这是初始问题:
            \n ------- \n
            {question} 
            \n ------- \n
            制定一个改进的问题: """,
                )
            ]
            
            model = ChatOpenAI(temperature=0, model=model_name)
            response = model.invoke(msg)
            return {"messages": [response]}
        
        def generate(state):
            """生成答案"""
            logger.info("生成答案")
            messages = state["messages"]
            question = messages[0].content
            docs = messages[-1].content
            
            prompt = PromptTemplate(
                template="""您是问答任务的助手。使用以下检索的上下文来回答问题。如果您不知道答案，就说您不知道。最多使用三个句子并保持答案简洁。
                问题: {question} 
                上下文: {context} 
                答案:""",
                input_variables=["context", "question"],
            )
            
            llm = ChatOpenAI(model_name=model_name, temperature=0)
            rag_chain = prompt | llm | StrOutputParser()
            
            response = rag_chain.invoke({"context": docs, "question": question})
            return {"messages": [response]}
        
        # 定义图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("agent", agent)
        retrieve = ToolNode(self.tools)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("rewrite", rewrite)
        workflow.add_node("generate", generate)
        
        # 添加边和逻辑
        workflow.add_edge(START, "agent")
        
        # 决定是否检索
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        
        # 检索后的边
        workflow.add_conditional_edges(
            "retrieve",
            grade_documents,
            {
                "generate": "generate",
                "rewrite": "rewrite"
            }
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")
        
        # 编译图
        self.graph = workflow.compile()
        logger.info("工作流图构建完成")
        return self.graph
    
    def run(self, query):
        """
        运行系统回答查询
        
        参数:
            query: 用户查询
        
        返回:
            回答内容
        """
        if not self.graph:
            raise ValueError("请先构建工作流图")
        
        logger.info(f"处理查询: {query}")
        
        # 准备输入
        inputs = {
            "messages": [
                HumanMessage(content=query),
            ]
        }
        
        # 执行
        response = self.graph.invoke(inputs)
        
        # 提取最终回答
        final_answer = response["messages"][-1]
        if hasattr(final_answer, 'content'):
            return final_answer.content
        else:
            return final_answer
    
    def stream_run(self, query):
        """
        流式运行系统回答查询，展示中间步骤
        
        参数:
            query: 用户查询
        
        返回:
            生成器，产生中间步骤和最终回答
        """
        if not self.graph:
            raise ValueError("请先构建工作流图")
        
        logger.info(f"流式处理查询: {query}")
        
        # 准备输入
        inputs = {
            "messages": [
                HumanMessage(content=query),
            ]
        }
        
        # 流式执行
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                yield {"node": key, "output": value}


def main():
    """示例用法"""
    # 设置您的OpenAI API密钥
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # 创建系统
    rag = AgenticRAG(openai_api_key)
    
    # 示例1：使用网页URL创建检索器
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
    ]
    # 示例2：使用PDF文件创建检索器（取消注释使用）
    # pdf_paths = ["./docs/sample1.pdf", "./docs/sample2.pdf"]
    # 示例3：使用PDF目录创建检索器（取消注释使用）
    # pdf_directory = "./pdf_docs"
    
    # 选择使用的数据源
    rag.create_retriever(urls=urls)
    # 或者使用PDF
    # rag.create_retriever(pdf_paths=pdf_paths)
    # 或者使用PDF目录
    # rag.create_retriever(pdf_directory=pdf_directory)
    # 或者混合使用
    # rag.create_retriever(urls=urls, pdf_paths=pdf_paths, pdf_directory=pdf_directory)
    
    # 构建图
    rag.build_graph(model_name="gpt-4o-mini")
    
    # 运行查询
    query = "什么是代理记忆的类型?"
    answer = rag.run(query)
    print(f"查询: {query}")
    print(f"回答: {answer}")
    
    # 流式运行示例
    print("\n流式运行示例:")
    query = "什么是提示工程中的思维链技术?"
    for step in rag.stream_run(query):
        print(f"节点: {step['node']}")
        print(f"输出: {step['output']}")
        print("-" * 50)


if __name__ == "__main__":
    main() 