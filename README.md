# 中文学术论文写作助手

中文学术论文写作助手是一个开源工具，可以自动化研究过程并生成符合中文学术规范的定制化论文。它允许您使用特定的模型、提示、论文结构和搜索工具来定制研究和写作过程。

![论文生成](https://github.com/user-attachments/assets/6595d5cd-c981-43ec-8e8b-209e4fefc596)

## 🚀 快速开始

确保您已设置所需搜索工具和模型的API密钥。

可用的搜索工具：

* [Tavily API](https://tavily.com/) - 通用网络搜索
* [Perplexity API](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api) - 通用网络搜索
* [Exa API](https://exa.ai/) - 强大的网络内容神经搜索
* [ArXiv](https://arxiv.org/) - 物理学、数学、计算机科学等领域的学术论文
* [PubMed](https://pubmed.ncbi.nlm.nih.gov/) - 来自MEDLINE、生命科学期刊和在线书籍的生物医学文献
* [Linkup API](https://www.linkup.so/) - 通用网络搜索
* [DuckDuckGo API](https://duckduckgo.com/) - 通用网络搜索
* [Google Search API/Scrapper](https://google.com/) - 创建自定义搜索引擎[在这里](https://programmablesearchengine.google.com/controlpanel/all)并获取API密钥[在这里](https://developers.google.com/custom-search/v1/introduction)

中文学术论文写作助手使用规划LLM来规划论文结构，并使用写作LLM来撰写论文：

* 您可以选择任何已与[`init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/)集成的模型
* 查看[这里](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)的支持集成完整列表

### 使用该包

```bash
pip install open-deep-research
```

如上所述，确保为LLM和搜索工具设置API密钥：
```bash
export TAVILY_API_KEY=<your_tavily_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
```

在Jupyter笔记本中的示例用法，参见[src/open_deep_research/graph.ipynb](src/open_deep_research/graph.ipynb)：

编译图：
```python
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

使用所需主题和配置运行图：
```python
import uuid 
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "tavily",
                           "planner_provider": "anthropic",
                           "planner_model": "claude-3-7-sonnet-latest",
                           "writer_provider": "anthropic",
                           "writer_model": "claude-3-5-sonnet-latest",
                           "max_search_depth": 1,
                           "knowledge_base_path": "./doc",  # 可选：指定知识库路径
                           }}

topic = "人工智能在医疗领域的应用与伦理考量"
async for event in graph.astream({"topic":topic,}, thread, stream_mode="updates"):
    print(event)
```

图将在生成论文计划后停止，您可以传递反馈来更新论文计划：
```python
from langgraph.types import Command
async for event in graph.astream(Command(resume="在章节中包含国内医疗AI发展现状的对比分析"), thread, stream_mode="updates"):
    print(event)
```

当您对论文计划满意后，可以传递`True`以继续生成论文：
```python
async for event in graph.astream(Command(resume=True), thread, stream_mode="updates"):
    print(event)
```

### 在本地运行LangGraph Studio UI

克隆仓库：
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
```

编辑`.env`文件，添加您的API密钥（例如，下面显示的是默认选择的API密钥）：
```bash
cp .env.example .env
```

根据需要设置模型和搜索工具的API。

以下是几个可用的模型和工具集成示例：
```bash
export TAVILY_API_KEY=<your_tavily_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
export OPENAI_API_KEY=<your_openai_api_key>
export PERPLEXITY_API_KEY=<your_perplexity_api_key>
export EXA_API_KEY=<your_exa_api_key>
export PUBMED_API_KEY=<your_pubmed_api_key>
export PUBMED_EMAIL=<your_email@example.com>
export LINKUP_API_KEY=<your_linkup_api_key>
export GOOGLE_API_KEY=<your_google_api_key>
export GOOGLE_CX=<your_google_custom_search_engine_id>
```

在本地启动带有LangGraph服务器的助手，它将在您的浏览器中打开：

#### Mac

```bash
# 安装uv包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖并启动LangGraph服务器
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

#### Windows / Linux

```powershell
# 安装依赖 
pip install -e .
pip install -U "langgraph-cli[inmem]" 

# 启动LangGraph服务器
langgraph dev
```

使用以下链接打开Studio UI：
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

(1) 提供一个`主题`并点击`Submit`：

<img width="1326" alt="input" src="https://github.com/user-attachments/assets/de264b1b-8ea5-4090-8e72-e1ef1230262f" />

(2) 这将生成一个论文计划并呈现给用户审阅。

(3) 我们可以传递一个字符串（`"..."`）作为反馈，根据反馈重新生成计划。

<img width="1326" alt="feedback" src="https://github.com/user-attachments/assets/c308e888-4642-4c74-bc78-76576a2da919" />

(4) 或者，我们可以直接传递`true`来接受计划。

<img width="1480" alt="accept" src="https://github.com/user-attachments/assets/ddeeb33b-fdce-494f-af8b-bd2acc1cef06" />

(5) 一旦接受，将生成论文章节。

<img width="1326" alt="report_gen" src="https://github.com/user-attachments/assets/74ff01cc-e7ed-47b8-bd0c-4ef615253c46" />

论文以markdown格式生成，并自动转换为HTML格式以便查看。

<img width="1326" alt="report" src="https://github.com/user-attachments/assets/92d9f7b7-3aea-4025-be99-7fb0d4b47289" />

## 📖 定制论文

您可以通过多个参数定制研究助手的行为：

- `report_structure`：为论文定义自定义结构（默认为标准的中文学术论文格式）
- `number_of_queries`：每个章节要生成的搜索查询数量（默认：2）
- `max_search_depth`：最大反思和搜索迭代次数（默认：2）
- `planner_provider`：规划阶段的模型提供商（默认："anthropic"，但可以是`init_chat_model`支持的任何提供商）
- `planner_model`：规划使用的具体模型（默认："claude-3-7-sonnet-latest"）
- `writer_provider`：写作阶段的模型提供商（默认："anthropic"，但可以是`init_chat_model`支持的任何提供商）
- `writer_model`：写作论文的模型（默认："claude-3-5-sonnet-latest"）
- `search_api`：用于网络搜索的API（默认："tavily"，选项包括"perplexity"、"exa"、"arxiv"、"pubmed"、"linkup"）
- `knowledge_base_path`：本地知识库路径（默认：`./doc`），用于存放PDF文档

这些配置允许您根据需要调整研究过程，从调整研究深度到为论文生成的不同阶段选择特定的AI模型。

### 新增功能

最新版本增加了多项功能，提高了论文写作的质量和用户体验：

#### 1. 智能搜索选择
- 系统会智能评估查询是需要Web搜索获取最新信息，还是可以从本地知识库获取
- 对于基础理论、历史发展等内容，优先使用知识库
- 对于时事、最新研究、数据等内容，优先使用Web搜索
- 可通过`knowledge_base_path`参数指定PDF文档库路径

#### 2. 内容质量评估系统
- 每个章节生成后会进行多维度质量评估
- 评估维度包括：内容完整性、学术规范、论证逻辑、表述准确性等
- 给出总体评分（100分制）和具体改进建议
- 对内容进行优缺点分析，提供明确的修改方向

#### 3. 内容自动修订功能
- 根据评估结果对不达标的章节内容进行自动修订
- 可设置质量阈值和最大修订次数，确保最终内容质量
- 每次修订后重新评估，直到达到质量标准或达到最大修订次数
- 修订过程完全自动化，无需人工干预

#### 4. HTML报告导出
- 自动将生成的论文转换为格式精美的HTML网页
- 输出到项目的`output`目录中，文件名包含主题和时间戳
- HTML版本更适合阅读和分享
- 保留了良好的排版和格式，支持参考文献交叉引用

### 搜索API配置

并非所有搜索API都支持其他配置参数。以下是支持的参数：

- **Exa**：`max_characters`、`num_results`、`include_domains`、`exclude_domains`、`subpages`
  - 注意：`include_domains`和`exclude_domains`不能一起使用
  - 当您需要将研究范围缩小到特定可信源、确保信息准确性或当您的研究需要使用指定域名（例如学术期刊、政府网站）时特别有用
  - 提供针对您特定查询定制的AI生成摘要，使从搜索结果中提取相关信息更容易
- **ArXiv**：`load_max_docs`、`get_full_documents`、`load_all_available_meta`
- **PubMed**：`top_k_results`、`email`、`api_key`、`doc_content_chars_max`
- **Linkup**：`depth`

带有Exa配置的示例：
```python
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "exa",
                           "search_api_config": {
                               "num_results": 5,
                               "include_domains": ["cnki.net", "sciencedirect.com"]
                           },
                           # 其他配置...
                           }}
```

### 模型注意事项

(1) 您可以传递任何已与[`init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/)集成的规划器和写作模型。查看[这里](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)的支持集成完整列表。

(2) **规划器和写作模型需要支持结构化输出**：在[此处](https://python.langchain.com/docs/integrations/chat/)查看您使用的模型是否支持结构化输出。

(3) 使用Groq时，如果您处于`on_demand`服务层级，则每分钟令牌数（TPM）有限制：
- `on_demand`服务层级的限制为`6000 TPM`
- 如果您想使用Groq模型进行章节写作，您会需要一个[付费计划](https://github.com/cline/cline/issues/47#issuecomment-2640992272)

(4) `deepseek-R1`[在函数调用方面不够强大](https://api-docs.deepseek.com/guides/reasoning_model)，而助手使用函数调用来生成论文章节和论文章节评分的结构化输出。在[这里](https://smith.langchain.com/public/07d53997-4a6d-4ea8-9a1f-064a85cd6072/r)查看示例跟踪。  
- 考虑使用在函数调用方面强大的提供商，如OpenAI、Anthropic和某些开源模型，如Groq的`llama-3.3-70b-versatile`。
- 如果您看到以下错误，可能是因为模型无法生成结构化输出（请参见[跟踪](https://smith.langchain.com/public/8a6da065-3b8b-4a92-8df7-5468da336cbe/r)）：
```
groq.APIError: Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.
```

## 工作原理
   
1. `规划与执行` - 中文学术论文写作助手遵循[规划与执行工作流程](https://github.com/assafelovic/gpt-researcher)，将规划与研究分开，允许在更耗时的研究阶段之前进行人工参与式批准论文计划。默认情况下，它使用[推理模型](https://www.youtube.com/watch?v=f0RbwrBcFmc)来规划论文章节。在此阶段，它使用网络搜索来收集有关论文主题的一般信息，以帮助规划论文章节。但它也接受用户提供的论文结构来帮助指导论文章节，以及对论文计划的人工反馈。
   
2. `研究与写作` - 论文的每个章节都是并行撰写的。研究助手首先通过评估决定从本地知识库还是通过网络搜索获取信息。对于网络搜索，它使用[Tavily API](https://tavily.com/)、[Perplexity](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api)、[Exa](https://exa.ai/)、[ArXiv](https://arxiv.org/)、[PubMed](https://pubmed.ncbi.nlm.nih.gov/)或[Linkup](https://www.linkup.so/)等工具。而对于知识库搜索，它使用向量检索技术从本地PDF文档中获取信息。

3. `质量保证` - 每个章节撰写完成后，系统会进行质量评估，给出评分和改进建议。如果质量低于设定阈值，系统会自动修订内容并重新评估，直到达到质量标准或达到最大修订次数。最终章节（如摘要和结论）在主体章节完成后撰写，以确保内容一致性。

4. `管理不同类型` - 中文学术论文写作助手基于LangGraph构建，它原生支持[使用助手](https://langchain-ai.github.io/langgraph/concepts/assistants/)进行配置管理。论文`结构`是图形配置中的一个字段，允许用户为不同类型的论文创建不同的助手。

5. `输出展示` - 系统将生成的论文以Markdown格式输出，并自动转换为HTML网页格式保存到output目录，便于阅读和分享。

## UX

### 本地部署

按照[快速开始](#-快速开始)在本地启动LangGraph服务器。

### 托管部署
 
您可以轻松部署到[LangGraph平台](https://langchain-ai.github.io/langgraph/concepts/#deployment-options)。
