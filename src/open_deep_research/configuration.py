import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict 

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

DEFAULT_REPORT_STRUCTURE = """使用以下结构创建关于用户提供主题的中文学术论文：

1. 论文标题
   - 准确、简洁地反映研究内容和方法

2. 摘要（无需研究）
   - 研究目的、方法、结果和结论的简要概述
   - 关键词：3-5个代表性关键词

3. 引言
   - 研究背景与意义
   - 国内外研究现状
   - 研究目的与方法

4. 正文部分：
   - 理论基础
   - 研究方法
   - 数据分析与结果
   - 讨论
   
5. 结论
   - 研究发现的摘要
   - 理论和实践意义
   - 研究局限性
   - 未来研究方向

6. 参考文献
   - 按照国标GB/T 7714-2015格式排列"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    report_structure: str = DEFAULT_REPORT_STRUCTURE # Defaults to the default report structure
    number_of_queries: int = 2 # Number of search queries to generate per iteration
    max_search_depth: int = 1 # Maximum number of reflection + search iterations
   #  planner_provider: str = "openai"    # Updated to Groq as provider
   #  planner_model: str = "deepseek/deepseek-chat-v3-0324:free"  
   #  writer_provider: str = "openai"     # Updated to Groq as provider
   #  writer_model: str = "o3-mini"   
    planner_provider: str = "deepseek"    # Updated to Groq as provider
    planner_model: str = "deepseek-chat"  
    writer_provider: str = "deepseek"     # Updated to Groq as provider
    writer_model: str = "deepseek-chat"   
   #  planner_provider: str = "google_genai"    
   #  planner_model: str = "gemini-2.0-flash"  
   #  writer_provider: str = "google_genai"    
   #  writer_model: str = "gemini-2.0-flash" 
    search_api: SearchAPI = SearchAPI.TAVILY # Default to TAVILY
    search_api_config: Optional[Dict[str, Any]] = None 

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
