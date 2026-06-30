# DeepSeek API 自动化评测系统

基于 DeepSeek 大模型 API 的 LLM 输出质量自动化评测系统。

[![GitHub](https://img.shields.io/badge/GitHub-bingu7%2Fai--test--project-blue)](https://github.com/bingu7/ai-test-project)

## 功能

- ✅ LLM 输出质量自动化评测（Faithfulness、Hallucination、Answer Relevancy 等）
- ✅ Prompt 模板对比测试（零样本/少样本/Chain-of-Thought）
- ✅ RAG 质量验证（LangChain + Chroma + Ragas）
- ✅ Allure 可视化测试报告
- ✅ GitHub Actions CI 集成

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 DEEPSEEK_API_KEY

# 运行测试
python run_all_tests.py
```

## 项目结构

```
ai-test-project/
├── tests/
│   ├── test_llm_output.py      # LLM 输出质量测试
│   ├── test_prompt_templates.py # Prompt 模板对比测试
│   └── test_rag_pipeline.py    # RAG 质量评估测试
├── config.py                   # 配置
├── run_all_tests.py            # 一键运行
└── requirements.txt            # 依赖
```
