from typing import Annotated, List, TypedDict, Literal, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import operator

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report.",
        default=True
    )
    content: str = Field(
        description="The content of the section.",
        default=""
    )
    evaluation: Optional[Union['ContentEvaluation', 'SimpleContentEvaluation']] = Field(
        description="章节内容的评估结果",
        default=None
    )

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    """搜索查询信息"""
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    """搜索查询列表"""
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class Feedback(BaseModel):
    """质量反馈信息"""
    grade: str = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )

class DimensionScore(BaseModel):
    """维度评分及评语"""
    score: int = Field(description="该维度的得分")
    comments: str = Field(description="该维度的评语")

class ContentEvaluation(BaseModel):
    """章节内容详细评估结果"""
    total_score: int = Field(description="总体评分(100分制)")
    dimension_scores: Dict[str, DimensionScore] = Field(description="各评估维度的得分和评语")
    strengths: List[str] = Field(description="内容优势")
    weaknesses: List[str] = Field(description="需要改进的方面")
    improvement_suggestions: List[str] = Field(description="具体修改建议")
    missing_content: List[str] = Field(description="缺失的内容")
    overall_assessment: str = Field(description="总体评价")

class SimpleContentEvaluation(BaseModel):
    """章节内容简单评估结果"""
    total_score: int = Field(description="总体评分(100分制)")
    strengths: List[str] = Field(description="内容优势(3-5点)")
    weaknesses: List[str] = Field(description="需要改进的方面(3-5点)")
    improvement_suggestions: List[str] = Field(description="具体修改建议")
    overall_assessment: str = Field(description="总体评价(200字左右)")

class ReportStateInput(TypedDict):
    topic: str # Report topic
    
class ReportStateOutput(TypedDict):
    final_report: str # Final report

class ReportState(TypedDict, total=False):
    topic: str 
    sections: List[Section]
    completed_sections: Annotated[List[Section], operator.add]  # 修改为与SectionState和SectionOutputState一致
    feedback_on_report_plan: Optional[str]
    report_sections_from_research: Optional[str]
    final_report: Optional[str]
    html_report: Optional[str]

class SectionState(TypedDict, total=False):
    topic: str
    section: Section 
    search_queries: List[SearchQuery]
    source_str: str
    search_iterations: int
    search_decision: str
    completed_sections: Annotated[List[Section], operator.add]  # 修改为与SectionOutputState相同的类型
    section_content_evaluation: Optional[Union[ContentEvaluation, SimpleContentEvaluation]]
    revision_count: int  # 修改次数计数器，用于防止死循环

class SectionOutputState(TypedDict, total=False):
    completed_sections: Annotated[List[Section], operator.add]  # 使用operator.add注解以支持多个值合并
