from typing import Literal
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from open_deep_research.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback,
    ContentEvaluation,
    SimpleContentEvaluation
)

from open_deep_research.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs,
    enhanced_query_writer_instructions,
    enhanced_report_planner_query_writer_instructions,
    enhanced_html_template_instructions,
    content_evaluator_instructions,
    enhanced_content_evaluator_instructions,
    section_revision_instructions,
    enhanced_section_revision_instructions
)

from open_deep_research.configuration import Configuration
from open_deep_research.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search
)

# 导入AgenticRAG相关模块
from open_deep_research.agentic_rag import AgenticRAG, create_tool_node, check_tool_calls
from pydantic import BaseModel, Field

## Nodes -- 

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """生成初始报告计划及其各个部分。
    
    此节点：
    1. 获取报告结构和搜索参数的配置
    2. 生成搜索查询以收集规划所需的背景信息
    3. 使用这些查询执行网络搜索
    4. 使用LLM生成带有结构的报告计划
    
    参数：
        state: 当前图形状态，包含报告主题
        config: 模型、搜索API等的配置
        
    返回：
        包含生成部分的字典
    """

    # 输入Topic是用户输入的报告主题  
    topic = state["topic"]
    feedback = state.get("feedback_on_report_plan", None)

    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # 获取配置字典，默认为空
    params_to_pass = get_search_params(search_api, search_api_config)  # 过滤参数

    # 如果需要，将JSON对象转换为字符串
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # 设置写作模型（用于查询写作的模型）
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    structured_llm = writer_model.with_structured_output(Queries)

    # 获取当前时间信息，用于时间敏感查询
    current_time = datetime.now()
    current_year = current_time.year
    current_month = current_time.month

    # 使用模板格式化系统指令
    system_instructions_query = enhanced_report_planner_query_writer_instructions.format(
        topic=topic, 
        report_organization=report_structure, 
        number_of_queries=number_of_queries,
        current_time=current_time.isoformat(),
        current_year=current_year,
        current_month=current_month
    )

    # 准备用于生成查询的用户消息
    user_message_query = f"""
    生成{number_of_queries}个高质量的搜索查询，以帮助规划关于以下主题的论文结构：

    主题：{topic}
    预期的论文组织：{report_structure}

    这些查询应该帮助收集有关如何有效构建此论文的信息，涵盖不同方面和视角。
    """

    # 使用增强提示生成查询
    results = structured_llm.invoke([
        SystemMessage(content=system_instructions_query),
        HumanMessage(content=user_message_query)
    ])

    # 网络搜索
    query_list = [query.search_query for query in results.queries]

    # 使用参数搜索网络
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    # 格式化系统指令
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure, context=source_str, feedback=feedback)

    # 设置规划器
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    # 报告规划器指令
    planner_message = """生成论文的章节。您的回答必须包含一个'sections'字段，其中包含章节列表。
                        每个章节必须具有：name, description, plan, research和content字段。"""

    # 运行规划器
    if planner_model == "claude-3-7-sonnet-latest":
        # 为claude-3-7-sonnet-latest作为规划器模型分配思考预算
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider, 
                                      max_tokens=20_000, 
                                      thinking={"type": "enabled", "budget_tokens": 16_000})

    else:
        # 对于其他模型，不特别分配思考令牌
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider)
    
    # 生成报告章节
    structured_llm = planner_llm.with_structured_output(Sections)
    report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections),
                                             HumanMessage(content=planner_message)])

    # 获取章节
    sections = report_sections.sections

    return {"sections": sections}

def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """获取关于报告计划的人类反馈并引导到下一步。
    此节点：
    1. 格式化当前报告计划以供人类审查
    2. 通过中断获取反馈
    3. 根据反馈引导到：
       - 如果计划被批准，则进行章节写作
       - 如果提供反馈，则重新生成计划
    
    参数：
        state: 当前图状态，包含待审查的章节
        config: 工作流的配置
        
    返回：
        命令以重新生成计划或开始章节写作
    """

    # Get sections
    topic = state["topic"]
    sections = state['sections']
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    # Get feedback on the report plan from interrupt
    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    
    feedback = interrupt(interrupt_message)

    # If the user approves the report plan, kick off section writing
    if isinstance(feedback, bool) and feedback is True:
        # Treat this as approve and kick off section writing
        return Command(goto=[
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ])
    
    # If the user provides feedback, regenerate the report plan 
    elif isinstance(feedback, str):
        # Treat this as feedback
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": feedback})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")

# 生成用于研究特定部分的搜索查询。
async def generate_queries(state: SectionState, config: RunnableConfig):
    """生成用于研究特定部分的搜索查询。
    
    该节点使用大型语言模型（LLM）根据部分主题和描述生成针对性的搜索查询。实现包括：
    1. 意图挖掘以理解用户更深层次的需求
    2. 多种认知视角以涵盖不同的角度
    3. 包含时间敏感的信息
    4. 通过适当的格式优化查询
    
    参数：
        state: 当前状态，包含部分详细信息
        config: 包含要生成的查询数量的配置
        
    返回：
        包含生成的搜索查询的字典
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Get current time information
    current_time = datetime.now()
    current_year = current_time.year
    current_month = current_time.month

    # Format system instructions using the template
    system_instructions = enhanced_query_writer_instructions.format(
        topic=topic, 
        section_topic=section.description, 
        number_of_queries=number_of_queries,
        current_time=current_time.isoformat(),
        current_year=current_year,
        current_month=current_month
    )

    print(system_instructions)

    # Prepare user message with any contextual information
    user_message = f"""
Generate {number_of_queries} high-quality search queries for researching:

Topic: {topic}
Section: {section.description}

These queries should cover different aspects and perspectives to gather comprehensive information for writing this section.
"""

    # Generate queries  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Generate queries with enhanced prompting
    queries = structured_llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=user_message)
    ])

    return {"search_queries": queries.queries}

def evaluate_query_source(state: SectionState, config: RunnableConfig) -> Literal["search_web", "search_knowledge_base"]:
    """评估查询并决定使用知识库还是Web搜索。
    
    此节点：
    1. 接收生成的搜索查询
    2. 使用LLM评估这些查询是否可以从本地知识库中获得，或需要最新的Web搜索
    3. 根据评估结果返回下一步操作
    
    参数：
        state: 当前状态，包含搜索查询
        config: 评估模型的配置
        
    返回：
        决策结果，指示下一步是进行Web搜索还是知识库检索
    """
    
    # 获取状态
    search_queries = state["search_queries"]
    topic = state["topic"]
    section = state["section"]
    
    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    
    # 获取评估模型
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    
    # 定义评分模型
    class QuerySource(BaseModel):
        """查询来源决策"""
        decision: str = Field(description="决策: 'web' 表示需要Web搜索以获取最新信息, 'kb' 表示可以从知识库获取")
        reasoning: str = Field(description="决策推理过程")
    
    # 初始化模型
    if planner_model == "claude-3-7-sonnet-latest":
        evaluator_model = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider,
            max_tokens=4000, 
            thinking={"type": "enabled", "budget_tokens": 2000}
        ).with_structured_output(QuerySource)
    else:
        evaluator_model = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider
        ).with_structured_output(QuerySource)
    
    # 准备查询内容
    query_list = [query.search_query for query in search_queries]
    query_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(query_list)])
    
    # 创建评估提示
    prompt_template = """您是一个专家顾问，负责决定应该在哪里搜索信息。
    
    您的任务是评估以下查询，并决定这些查询是：
    1. 需要通过Web搜索获取最新、时效性强的信息（选择'web'）
    2. 可以通过现有知识库获得足够的信息（选择'kb'）
    
    主题：{topic}
    章节：{section_name} - {section_description}
    
    查询列表：
    {queries}
    
    以下情况应选择'web'：
    - 需要最新的数据、统计或事件信息
    - 涉及当前趋势或技术发展
    - 需要最新的学术研究结果
    - 涉及时事或近期发展
    
    以下情况应选择'kb'：
    - 概念解释、基础理论或历史发展
    - 已经确立的方法或技术
    - 经典案例研究或文献
    - 不太可能在短期内发生重大变化的信息
    
    请分析查询列表并给出决策：'web'或'kb'。
    """
    
    # 格式化提示
    formatted_prompt = prompt_template.format(
        topic=topic,
        section_name=section.name,
        section_description=section.description,
        queries=query_text
    )
    
    # 获取评估结果
    evaluation = evaluator_model.invoke([
        SystemMessage(content=formatted_prompt),
        HumanMessage(content="请分析以上查询并决定是使用Web搜索还是知识库检索。")
    ])
    
    # 记录决策理由
    print(f"查询源决策理由: {evaluation.reasoning}")
    
    # 检测函数是作为条件边还是作为节点调用
    # 作为条件边使用时会传递一个特殊的'__run_as_condition'配置
    is_condition = config.get("__run_as_condition", False) if config else False
    
    if is_condition:
        # 作为条件边运行时，返回下一个节点名称
        if evaluation.decision.lower() == "web":
            return "search_web" 
        else:
            return "search_knowledge_base"
    else:
        # 作为节点运行时，返回状态更新的字典
        decision = "search_web" if evaluation.decision.lower() == "web" else "search_knowledge_base"
        # 返回经过决策的结果，并将决策存储在状态中
        return {"search_decision": decision}

async def search_knowledge_base(state: SectionState, config: RunnableConfig):
    """从知识库中检索与查询相关的信息。
    
    此节点：
    1. 获取生成的查询
    2. 使用AgenticRAG从知识库中检索相关信息
    3. 将结果格式化为可用的上下文
    
    参数：
        state: 当前状态，包含搜索查询
        config: 检索配置
        
    返回：
        包含检索结果和更新的迭代计数的字典
    """
        # 获取配置
    configurable = Configuration.from_runnable_config(config)
    
    # 获取评估模型
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model = get_config_value(configurable.writer_model)

    # 获取状态
    search_queries = state["search_queries"]
    query_list = [query.search_query for query in search_queries]
    
    # 初始化AgenticRAG
    rag = AgenticRAG(model_name=writer_model, model_provider=writer_provider)
    
    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    
    # 配置知识库路径
    pdf_directory = configurable.knowledge_base_path or "./doc"
    
    # 创建检索器
    try:
        rag.create_retriever(pdf_directory=pdf_directory)
    except ValueError:
        # 如果没有找到文档，尝试从Web获取
        print("知识库中没有找到文档，正在切换到Web搜索...")
        # 执行Web搜索作为后备
        search_api = get_config_value(configurable.search_api)
        search_api_config = configurable.search_api_config or {}
        params_to_pass = get_search_params(search_api, search_api_config)
        source_str = await select_and_execute_search(search_api, query_list, params_to_pass)
        return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}
    
    # 构建图
    model_name = get_config_value(configurable.writer_model)
    rag.build_graph(model_name=model_name)
    
    # 合并查询以获得更全面的结果
    combined_query = " ".join(query_list)
    
    # 执行检索
    result = rag.run(combined_query)
    
    return {"source_str": result, "search_iterations": state["search_iterations"] + 1}

async def search_web(state: SectionState, config: RunnableConfig):
    """执行该部分查询的网络搜索。
    
    此节点：
    1. 获取生成的查询
    2. 使用配置的搜索API执行搜索
    3. 将结果格式化为可用的上下文
    
    参数：
        state: 当前状态，包含搜索查询
        config: 搜索API配置
        
    返回：
        包含搜索结果和更新的迭代计数的字典
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal["evaluate_query_source", "evaluate_section_content"]]:
    """撰写报告的一部分并评估是否需要更多研究。
    
    此节点：
    1. 使用搜索结果撰写部分内容
    2. 评估该部分的质量
    3. 要么：
       - 如果质量通过，进入内容评估阶段
       - 如果质量不合格，则触发更多研究和评估
    
    参数：
        state: 当前状态，包含搜索结果和部分信息
        config: 用于撰写和评估的配置
        
    返回：
        命令以完成部分、进入内容评估或评估查询源
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Format system instructions
    section_writer_inputs_formatted = section_writer_inputs.format(topic=topic, 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str, 
                                                             section_content=section.content)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 

    section_content = writer_model.invoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    
    # Write content to the section object  
    section.content = section_content.content

    # Grade prompt 
    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_instructions_formatted = section_grader_instructions.format(topic=topic, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

    # Use planner model for reflection
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, 
                                           max_tokens=20_000, 
                                           thinking={"type": "enabled", "budget_tokens": 16_000}).with_structured_output(Feedback)
    else:
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider).with_structured_output(Feedback)
    # Generate feedback
    feedback = reflection_model.invoke([SystemMessage(content=section_grader_instructions_formatted),
                                        HumanMessage(content=section_grader_message)])

    # If the section is passing or the max search depth is reached, proceed to content evaluation
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Proceed to content evaluation
        return Command(
            update={"section": section},
            goto="evaluate_section_content"
        )

    # Update the existing section with new content and update search queries
    else:
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": section},
            goto="evaluate_query_source"
        )

def evaluate_section_content(state: SectionState, config: RunnableConfig) -> Command[Literal["revise_section_content", END]]:
    """对章节内容进行详细质量评估。
    
    此节点：
    1. 使用高级评估模型对章节内容进行全面评估
    2. 生成多个维度的评分和建议
    3. 将评估结果保存到章节中
    4. 根据评分决定是否需要修改内容
    
    参数：
        state: 当前状态，包含章节内容
        config: 用于内容评估的配置
        
    返回：
        命令以修改内容或完成章节
    """
    # 获取状态
    topic = state["topic"]
    section = state["section"]
    
    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    
    # 初始化修改计数器（如果不存在）
    revision_count = state.get("revision_count", 0)
    
    # 设置评估提示
    evaluation_prompt_formatted = enhanced_content_evaluator_instructions.format(
        topic=topic,
        section_name=section.name,
        section_topic=section.description,
        section_content=section.content
    )
    
    # 使用planner模型进行评估
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    
    # 生成评估结果
    evaluation_message = "请对提供的章节内容进行全面评估，根据给定的标准提供详细的质量评价和改进建议。"
    
    try:
        # 尝试使用文本输出格式进行评估，而不是结构化输出
        evaluation_model = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider, 
            max_tokens=20_000, 
            thinking={"type": "enabled", "budget_tokens": 16_000}
        )
        
        evaluation_result_text = evaluation_model.invoke([
            SystemMessage(content=evaluation_prompt_formatted),
            HumanMessage(content=evaluation_message)
        ])
        
        # 手动解析评估结果文本
        result_text = evaluation_result_text.content
        
        # 创建简化版评估结果对象
        evaluation_result = SimpleContentEvaluation(
            total_score=extract_total_score(result_text),
            strengths=extract_list_items(result_text, "内容优势"),
            weaknesses=extract_list_items(result_text, "内容不足"),
            improvement_suggestions=extract_list_items(result_text, "改进建议"),
            overall_assessment=extract_overall_assessment(result_text)
        )
        
        # 将评估结果保存到章节中
        section.evaluation = evaluation_result
        
    except Exception as e:
        print(f"内容评估过程中出现错误: {str(e)}")
        print("尝试使用简化版评估模型...")
        
        # 使用更简化的评估方法
        simple_evaluation_prompt = content_evaluator_instructions.format(
            topic=topic,
            section_name=section.name,
            section_topic=section.description,
            section_content=section.content
        )
        
        simple_evaluation_model = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider
        )
        
        try:
            result_text = simple_evaluation_model.invoke([
                SystemMessage(content=simple_evaluation_prompt),
                HumanMessage(content=evaluation_message)
            ]).content
            
            # 基于文本创建简化版评估结果
            evaluation_result = SimpleContentEvaluation(
                total_score=extract_total_score(result_text),
                strengths=extract_list_items(result_text, "优势"),
                weaknesses=extract_list_items(result_text, "需要改进"),
                improvement_suggestions=extract_list_items(result_text, "修改建议"),
                overall_assessment=extract_overall_assessment(result_text)
            )
            
            # 确保至少有一些内容，如果解析失败
            if not evaluation_result.strengths:
                evaluation_result.strengths = ["内容结构清晰"]
            if not evaluation_result.weaknesses:
                evaluation_result.weaknesses = ["需要补充更多细节"]
            if not evaluation_result.improvement_suggestions:
                evaluation_result.improvement_suggestions = ["添加更多专业术语和概念解释"]
            if not evaluation_result.overall_assessment:
                evaluation_result.overall_assessment = "章节内容基本符合学术要求，但仍有改进空间。"
            
            # 将评估结果保存到章节中
            section.evaluation = evaluation_result
            
        except Exception as e2:
            print(f"简化评估也失败了: {str(e2)}")
            # 创建一个基本的评估结果，确保流程可以继续
            section.evaluation = SimpleContentEvaluation(
                total_score=75,
                strengths=["内容结构清晰", "主题相关性强", "术语使用准确"],
                weaknesses=["深度不足", "论证不够充分", "缺少一些关键概念"],
                improvement_suggestions=["增加内容深度", "补充更多学术引用", "加强论证"],
                overall_assessment="章节基本覆盖了主题，但需要进一步完善和深化内容。"
            )
    
    # 为控制台打印评估摘要
    print(f"章节 '{section.name}' 评估完成")
    print(f"总分: {section.evaluation.total_score}/100")
    print(f"优势: {', '.join(section.evaluation.strengths[:3])}")
    print(f"改进建议: {', '.join(section.evaluation.improvement_suggestions[:3])}")
    
    # 根据评分决定是否需要修改
    # 设置修改阈值，低于此分数需要修改
    revision_threshold = 85
    max_revisions = 3  # 最大修改次数
    
    if section.evaluation.total_score < revision_threshold and revision_count < max_revisions:
        # 需要修改内容
        return Command(
            update={"section_content_evaluation": section.evaluation, "revision_count": revision_count},
            goto="revise_section_content"
        )
    else:
        # 内容质量已达标或已达到最大修改次数，完成章节
        if revision_count > 0:
            print(f"章节 '{section.name}' 已完成 {revision_count} 轮修改，最终得分: {section.evaluation.total_score}/100")
        # 使用Command格式返回，与状态注解兼容
        return Command(
            update={"completed_sections": [section]},
            goto=END
        )

# 辅助函数：从文本中提取总分
def extract_total_score(text):
    import re
    # 尝试找到总分的模式
    patterns = [
        r"总分[:：]\s*(\d+)",
        r"总分[为是]\s*(\d+)",
        r"总体评分[:：]\s*(\d+)",
        r"总分[为是]\s*(\d+)/100",
        r"(\d+)[分]"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
            except:
                pass
    
    # 如果无法提取，返回默认分数
    return 75

# 辅助函数：从文本中提取列表项
def extract_list_items(text, section_name):
    import re
    
    # 尝试找到章节部分
    section_pattern = f"{section_name}[：:](.*?)(?:\n\n|\n\d+\.|\n[^\n]+[:：]|$)"
    section_match = re.search(section_pattern, text, re.DOTALL)
    
    if section_match:
        section_text = section_match.group(1).strip()
        
        # 提取列表项，可能以数字、破折号或星号开头
        items = re.findall(r'(?:^|\n)[\d*\-•]+[.、)：:\s]+(.+?)(?=(?:\n[\d*\-•]+[.、)：:\s]+|\n\n|\n[^\n]+[:：]|$))', '\n' + section_text, re.DOTALL)
        
        # 如果没有找到格式化的列表项，尝试按行分割
        if not items:
            items = [line.strip() for line in section_text.split('\n') if line.strip()]
        
        # 清理项目
        cleaned_items = []
        for item in items:
            # 移除可能的前导编号/符号
            clean_item = re.sub(r'^[\d*\-•]+[.、)：:\s]+', '', item).strip()
            if clean_item:
                cleaned_items.append(clean_item)
        
        return cleaned_items[:5]  # 最多返回5项
    
    return []  # 默认返回空列表

# 辅助函数：提取总体评价
def extract_overall_assessment(text):
    import re
    
    # 尝试找到总体评价部分
    patterns = [
        r"总体评价[：:](.*?)(?:\n\n|\n[^\n]+[:：]|$)",
        r"整体评价[：:](.*?)(?:\n\n|\n[^\n]+[:：]|$)",
        r"总体评估[：:](.*?)(?:\n\n|\n[^\n]+[:：]|$)",
        r"总结[：:](.*?)(?:\n\n|\n[^\n]+[:：]|$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # 如果找不到明确的总体评价部分，尝试提取最后一段
    paragraphs = text.split('\n\n')
    if paragraphs:
        return paragraphs[-1].strip()
    
    return "章节内容基本符合要求，但仍有改进空间。"  # 默认评价

def revise_section_content(state: SectionState, config: RunnableConfig) -> Command[Literal["evaluate_section_content"]]:
    """根据评估结果修改章节内容。
    
    此节点：
    1. 根据评估反馈修改章节内容
    2. 增加修改计数器
    3. 返回到评估节点重新评估修改后的内容
    
    参数：
        state: 当前状态，包含章节内容和评估结果
        config: 用于内容修改的配置
        
    返回：
        命令以重新评估修改后的内容
    """
    # 获取状态
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]
    evaluation = section.evaluation
    
    # 获取修改次数（默认为0）
    revision_count = state.get("revision_count", 0) + 1
    print(f"开始第 {revision_count} 轮修改章节 '{section.name}'")
    
    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    
    # 准备修改提示的内容
    try:
        # 尝试使用增强版提示（适用于ContentEvaluation）
        dimension_scores_text = "\n".join([
            f"{key}: {value.score}/100 - {value.comments}" 
            for key, value in evaluation.dimension_scores.items()
        ])
        strengths_text = "\n".join([f"- {s}" for s in evaluation.strengths])
        weaknesses_text = "\n".join([f"- {s}" for s in evaluation.weaknesses])
        suggestions_text = "\n".join([f"- {s}" for s in evaluation.improvement_suggestions])
        missing_text = "\n".join([f"- {s}" for s in evaluation.missing_content])
        
        # 使用增强版修改提示
        revision_prompt = enhanced_section_revision_instructions.format(
            topic=topic,
            section_name=section.name,
            section_topic=section.description,
            section_content=section.content,
            total_score=evaluation.total_score,
            dimension_scores=dimension_scores_text,
            strengths=strengths_text,
            weaknesses=weaknesses_text,
            improvement_suggestions=suggestions_text,
            missing_content=missing_text,
            overall_assessment=evaluation.overall_assessment,
            source_str=source_str,
            revision_count=revision_count
        )
    except AttributeError:
        # 使用简化版提示（适用于SimpleContentEvaluation）
        strengths_text = "\n".join([f"- {s}" for s in evaluation.strengths])
        weaknesses_text = "\n".join([f"- {s}" for s in evaluation.weaknesses])
        suggestions_text = "\n".join([f"- {s}" for s in evaluation.improvement_suggestions])
        
        # 使用基础版修改提示
        revision_prompt = section_revision_instructions.format(
            topic=topic,
            section_name=section.name,
            section_topic=section.description,
            section_content=section.content,
            total_score=evaluation.total_score,
            strengths=strengths_text,
            weaknesses=weaknesses_text,
            improvement_suggestions=suggestions_text,
            missing_content="",  # 简化版评估没有缺失内容字段
            overall_assessment=evaluation.overall_assessment,
            source_str=source_str
        )
    
    # 使用写作模型修改内容
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    
    writer_model = init_chat_model(
        model=writer_model_name, 
        model_provider=writer_provider
    )
    
    # 生成修改后的内容
    revised_content = writer_model.invoke([
        SystemMessage(content=revision_prompt),
        HumanMessage(content="请根据评估反馈修改章节内容。")
    ])
    
    # 更新章节内容
    section.content = revised_content.content
    
    # 返回到评估节点重新评估
    return Command(
        update={"section": section, "revision_count": revision_count},
        goto="evaluate_section_content"
    )

# 下一个流程   
def write_final_sections(state: SectionState, config: RunnableConfig):
    """为不需要研究的章节（比如摘要、引言、结论等）撰写内容。
    
    此节点：
    1. 接收一个不需要研究的章节
    2. 使用已完成的研究章节作为上下文
    3. 撰写该章节内容
    
    参数：
        state: 当前状态，包含章节和上下文
        config: 运行时配置
        
    返回：
        包含一个已完成章节的结果
    """

    # 获取状态
    topic = state["topic"]
    section = state["section"]
    report_sections = state["report_sections_from_research"]
    
    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    
    # 根据章节类型设置提示模板
    prompt_formatted = final_section_writer_instructions.format(
        topic=topic,
        section_name=section.name,
        section_topic=section.description,
        context=report_sections
    )
    
    # 使用生成器模型撰写章节
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    
    # 生成章节内容
    section_content = writer_model.invoke([
        SystemMessage(content=prompt_formatted),
        HumanMessage(content=f"请撰写{section.name}章节。")
    ])
    
    # 更新章节内容
    section.content = section_content.content

    # 使用Command格式返回，与状态注解兼容
    return Command(
        update={"completed_sections": [section]},
        goto=END
    )

def gather_completed_sections(state: ReportState):
    # 格式化已完成的章节作为写作最终章节的上下文。
    #
    # 此节点将所有已完成的研究章节格式化为一个
    # 单一的上下文字符串，以便于写作总结章节。
    #
    # 参数：
    #     state: 当前状态，包含已完成的章节
    #
    # 返回：
    #     包含格式化章节作为上下文的字典

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}

def compile_final_report(state: ReportState, config: RunnableConfig):
    """合并所有章节到最终报告中并生成网页展示。
    
    此节点：
    1. 获取所有已完成的章节
    2. 按照原始计划排序
    3. 处理参考文献，确保不重复且按序号排列
    4. 将所有内容组合成最终报告
    5. 使用大语言模型生成HTML网页展示报告
    
    参数：
        state: 包含所有已完成章节的当前状态
        config: 运行时配置
        
    返回：
        包含完整报告和HTML展示的字典
    """

    # 获取章节
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # 更新章节内容，同时保持原始顺序
    for section in sections:
        section.content = completed_sections[section.name]

    # 提取所有参考文献并消除重复
    all_references = []
    reference_urls = set()
    ref_map = {}  # 用于映射原引用编号到新引用编号
    
    # 第一遍：收集所有唯一参考文献
    for section in sections:
        content = section.content
        if "### 参考文献" in content:
            # 分割内容和参考文献
            content_parts = content.split("### 参考文献")
            if len(content_parts) > 1:
                refs_text = content_parts[1].strip()
                # 解析参考文献行
                for ref_line in refs_text.split('\n'):
                    ref_line = ref_line.strip()
                    if ref_line and ref_line.startswith('[') and ']:' in ref_line:
                        # 提取URL
                        url_part = ref_line.split(':', 1)[1].strip() if ':' in ref_line else ""
                        if url_part and url_part not in reference_urls:
                            reference_urls.add(url_part)
                            all_references.append(ref_line)
    
    # 重新编号参考文献
    numbered_references = []
    for i, ref in enumerate(all_references, 1):
        # 提取原引用编号和引用内容
        old_num = ref.split(']')[0][1:].strip()
        ref_content = ref.split(']', 1)[1].strip()
        # 创建新的引用行并存储映射关系
        new_ref = f"[{i}]{ref_content}"
        ref_map[old_num] = str(i)
        numbered_references.append(new_ref)
    
    # 第二遍：更新每个章节中的引用编号
    for section in sections:
        content = section.content
        if "### 参考文献" in content:
            # 分割内容和参考文献
            content_parts = content.split("### 参考文献")
            main_content = content_parts[0]
            
            # 更新正文中的引用编号
            for old_num, new_num in ref_map.items():
                main_content = main_content.replace(f"[{old_num}]", f"[{new_num}]")
            
            # 移除原参考文献部分，因为会在最后统一添加
            section.content = main_content.rstrip()
    
    # 组合最终报告
    formatted_sections = "\n\n".join([s.content for s in sections])
    
    # 添加统一的参考文献部分
    if numbered_references:
        references_section = "### 参考文献\n" + "\n".join(numbered_references)
        final_report = f"{formatted_sections}\n\n{references_section}"
    else:
        final_report = formatted_sections
    
    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    model_provider = get_config_value(configurable.Web_provider)
    model_name = get_config_value(configurable.Web_model)
    
    # 运行规划器
    if model_name == "claude-3-7-sonnet-latest":
        # 为claude-3-7-sonnet-latest作为规划器模型分配思考预算
        planner_llm = init_chat_model(model=model_name, 
                                      model_provider=model_provider, 
                                      max_tokens=20_000, 
                                      thinking={"type": "enabled", "budget_tokens": 16_000})
    else:
        # 对于其他模型，不特别分配思考令牌
        planner_llm = init_chat_model(model=model_name, 
                                      model_provider=model_provider)
    
    # 创建提示模板来生成HTML
    html_prompt = PromptTemplate(
        template=enhanced_html_template_instructions,
        input_variables=["report"]
    )
    
    # 生成HTML网页
    html_chain = html_prompt | planner_llm | StrOutputParser()
    html_output = html_chain.invoke({"report": final_report})
    
    # 将HTML保存到文件
    import os
    from datetime import datetime
    
    # 创建输出目录
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic = state.get("topic", "report").replace(" ", "_")
    html_filename = f"{topic}_{timestamp}.html"
    html_path = os.path.join(output_dir, html_filename)
    
    # 保存HTML文件
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_output)
    
    return {"final_report": final_report, "html_report": html_path}

def initiate_final_section_writing(state: ReportState):
    """为撰写不需要研究的部分创建并行任务。
    
    此边缘函数识别不需要研究的章节，并为每个章节创建并行写作任务。
    
    参数：
        state: 当前状态，包含所有章节和研究上下文
        
    返回：
        用于并行章节写作的 Send 命令列表
    """

    # 使用 Send() API 并行启动不需要研究的章节写作任务
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("evaluate_query_source", evaluate_query_source)
section_builder.add_node("search_web", search_web)
section_builder.add_node("search_knowledge_base", search_knowledge_base)
section_builder.add_node("write_section", write_section)
section_builder.add_node("evaluate_section_content", evaluate_section_content)
section_builder.add_node("revise_section_content", revise_section_content)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "evaluate_query_source")

# 定义一个条件函数，使用search_decision状态字段进行路由
def route_by_search_decision(state):
    """基于评估决策路由到适当的搜索节点"""
    return state.get("search_decision", "search_web")  # 默认为web搜索

# 添加条件边，基于search_decision字段决定搜索路径
section_builder.add_conditional_edges(
    "evaluate_query_source",
    route_by_search_decision,
    {
        "search_web": "search_web",
        "search_knowledge_base": "search_knowledge_base"
    }
)

section_builder.add_edge("search_web", "write_section")
section_builder.add_edge("search_knowledge_base", "write_section")
section_builder.add_edge("revise_section_content", "evaluate_section_content")

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
