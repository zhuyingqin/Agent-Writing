from typing import Literal
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

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
    Feedback
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
    enhanced_report_planner_query_writer_instructions
)

from open_deep_research.configuration import Configuration
from open_deep_research.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search
)

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
    """Get human feedback on the report plan and route to next steps.
    
    This node:
    1. Formats the current report plan for human review
    2. Gets feedback via an interrupt
    3. Routes to either:
       - Section writing if plan is approved
       - Plan regeneration if feedback is provided
    
    Args:
        state: Current graph state with sections to review
        config: Configuration for the workflow
        
    Returns:
        Command to either regenerate plan or start section writing
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
    
def generate_queries(state: SectionState, config: RunnableConfig):
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

async def search_web(state: SectionState, config: RunnableConfig):
    """Execute web searches for the section queries.
    
    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context
    
    Args:
        state: Current state with search queries
        config: Search API configuration
        
    Returns:
        Dict with search results and updated iteration count
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

def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report and evaluate if more research is needed.
    
    This node:
    1. Writes section content using search results
    2. Evaluates the quality of the section
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails
    
    Args:
        state: Current state with search results and section info
        config: Configuration for writing and evaluation
        
    Returns:
        Command to either complete section or do more research
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

    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Publish the section to completed sections 
        return  Command(
        update={"completed_sections": [section]},
        goto=END
    )

    # Update the existing section with new content and update search queries
    else:
        return  Command(
        update={"search_queries": feedback.follow_up_queries, "section": section},
        goto="search_web"
        )

# 下一个流程   
def write_final_sections(state: SectionState, config: RunnableConfig):
    """使用已完成的章节作为上下文，编写不需要研究的章节。
    
    此节点处理诸如结论或摘要等章节，这些章节基于
    已研究的章节，而不是直接需要研究。
    
    参数：
        state: 当前状态，包含已完成章节的上下文
        config: 写作模型的配置
        
    返回：
        包含新编写章节的字典
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get state 
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    
    section_content = writer_model.invoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to section 
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

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

def compile_final_report(state: ReportState):
    """合并所有章节到最终报告中。
    
    此节点：
    1. 获取所有已完成的章节
    2. 按照原始计划排序
    3. 处理参考文献，确保不重复且按序号排列
    4. 将所有内容组合成最终报告
    
    参数：
        state: 包含所有已完成章节的当前状态
        
    返回：
        包含完整报告的字典
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

    return {"final_report": final_report}

def initiate_final_section_writing(state: ReportState):
    """Create parallel tasks for writing non-research sections.
    
    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one.
    
    Args:
        state: Current state with all sections and research context
        
    Returns:
        List of Send commands for parallel section writing
    """

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

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
