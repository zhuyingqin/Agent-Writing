# Deep Research 增强功能实现

## 新增功能

1. 文件检索功能：能够从本地文件中检索相关内容
2. 代码执行功能：能够生成并执行与研究主题相关的Python代码

## 文件修改

1. state.py - 添加了文件检索和代码执行相关的状态类型
2. configuration.py - 添加了文件检索和代码执行相关的配置选项
3. file_code_utils.py - 实现了文件检索和代码执行的核心功能
4. graph.py - 修改了图的结构，添加了新的节点和边
5. demo_enhanced_research.py - 提供了使用增强功能的演示

## 使用方法

1. 启用文件检索：设置 enable_file_retrieval=True 并配置 file_retrieval_paths
2. 启用代码执行：设置 enable_code_execution=True
3. 运行演示：python src/open_deep_research/demo_enhanced_research.py

