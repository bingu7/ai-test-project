# DeepSeek API 自动化评测系统

基于 DeepSeek 大模型 API 的 LLM 输出质量自动化评测系统。

[![GitHub](https://img.shields.io/badge/GitHub-bingu7%2Fai--test--project-blue)](https://github.com/bingu7/ai-test-project)

## 功能

- ✅ LLM 输出质量自动化评测（指令遵循、事实一致、安全过滤、边界测试、中文理解）
- ✅ Prompt 模板对比测试（零样本 / 少样本 / Chain-of-Thought）
- ✅ RAG 质量验证（知识库检索 + Ragas 评估）
- ✅ 无 API Key 时自动 Mock 运行，有 Key 时调用真实模型
- ✅ HTML 测试报告（需安装 pytest-html）

## 快速开始

```bash
# 1️⃣ 安装依赖
pip install -r requirements.txt

# 2️⃣ 配置 API Key（可选，无 Key 会自动走 Mock 模式）
notepad .env
# 在里面填入:
# DEEPSEEK_API_KEY=sk-your-key-here
# DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 3️⃣ 一键运行全部测试
python run_all_tests.py

# 或单独运行某个模块
python -m pytest tests/test_llm_output.py -v

# 生成 HTML 报告
pip install pytest-html
python run_all_tests.py
```

> 💡 **无 API Key 也能跑** — 项目自动检测 Key，没有就 Mock 返回，展示框架和报告流程。

## 项目结构

```
ai-test-project/
├── .env                     # API Key 配置（不上传 GitHub）
├── .env.example             # 配置模板
├── requirements.txt         # 依赖列表
├── config.py                # 配置加载
├── run_all_tests.py         # 一键运行入口
├── README.md
├── .gitignore
├── tests/
│   ├── test_llm_output.py       # LLM 输出质量测试（7个用例）
│   ├── test_prompt_templates.py # Prompt 策略对比测试（3个用例）
│   └── test_rag_pipeline.py    # RAG 知识库测试（4个用例）
└── reports/                    # HTML 测试报告（运行后生成）
```

## 测试项说明

| 文件 | 测试内容 | 用例数 |
|------|---------|--------|
| `test_llm_output.py` | 指令遵循、格式约束、事实一致、安全过滤、空输入、长文本、中文理解 | 7+ |
| `test_prompt_templates.py` | Zero-shot / Few-shot / Chain-of-Thought 三种策略对比 | 3 |
| `test_rag_pipeline.py` | 知识库完整、文档查询设计、LangChain 检索、Ragas 评估 | 4 |

## 项目亮点（面试时可用）

- **Mock 机制**：无 Key 也能完整演示测试框架，面试官不需要环境就能看懂
- **多维覆盖**：从功能性（指令遵循）到安全性（拒绝有害请求）到边界（空/长输入）
- **Prompt 对比**：量化不同策略对输出的影响，体现测试设计深度
- **RAG 质量**：检索准确率 + 生成质量双重验证
- **可扩展**：换 Key / 加测试用例非常容易
