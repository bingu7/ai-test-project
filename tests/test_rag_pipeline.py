"""RAG pipeline quality evaluation using LangChain + Chroma + Ragas."""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()

USE_MOCK = not os.getenv("DEEPSEEK_API_KEY") or "..." in (os.getenv("DEEPSEEK_API_KEY") or "")

# Mock knowledge base
SAMPLE_DOCUMENTS = [
    "Python 是一种广泛使用的高级编程语言，由 Guido van Rossum 于 1991 年首次发布。",
    "Python 的设计哲学强调代码的可读性和简洁的语法，尤其是使用空格缩进划分代码块。",
    "Python 是动态类型语言，支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。",
    "Python 拥有一个庞大且活跃的社区，提供了大量的第三方库和框架。",
    "Python 广泛应用于 Web 开发、数据分析、人工智能、科学计算和自动化运维等领域。",
    "Python 的包管理工具 pip 可以方便地安装和管理第三方库。",
    "Python 3 是当前的主要版本，Python 2 已于 2020 年停止维护。",
    "Python 的虚拟环境（virtualenv/venv）可以隔离项目的依赖关系。",
]

# Try langchain
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Try ragas
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def test_sample_documents_loaded():
    """Verify the test knowledge base is intact."""
    assert len(SAMPLE_DOCUMENTS) == 8
    assert "Python" in SAMPLE_DOCUMENTS[0]
    print(f"\n[Knowledge Base] {len(SAMPLE_DOCUMENTS)} documents loaded")


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain not installed")
def test_rag_retrieval():
    """Test that RAG retrieval returns relevant documents. Requires HF model download."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception:
        pytest.skip("Could not load embedding model (check internet/proxy)")
    docs = [Document(page_content=t) for t in SAMPLE_DOCUMENTS]
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    test_cases = [
        ("Python 是什么时候发布的？", "1991"),
        ("Python 的设计哲学是什么？", "可读性"),
        ("Python 的包管理工具是什么？", "pip"),
    ]

    for query, expected in test_cases:
        results = retriever.invoke(query)
        print(f"\n[RAG] Query: {query}")
        for i, r in enumerate(results):
            print(f"  [{i+1}] {r.page_content[:80]}")
        combined = " ".join([r.page_content for r in results])
        assert expected in combined, f"Expected '{expected}' in results for '{query}'"


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
def test_ragas_evaluation():
    """Evaluate RAG quality with Ragas metrics."""
    from datasets import Dataset

    data = {
        "question": [
            "Python 是什么时候发布的？",
            "Python 用于哪些领域？",
        ],
        "answer": [
            "Python 由 Guido van Rossum 于 1991 年首次发布。",
            "Python 广泛应用于 Web 开发、数据分析、人工智能和科学计算等领域。",
        ],
        "contexts": [
            ["Python 是一种广泛使用的高级编程语言，由 Guido van Rossum 于 1991 年首次发布。"],
            [
                "Python 广泛应用于 Web 开发、数据分析、人工智能、科学计算和自动化运维等领域。",
                "Python 拥有一个庞大且活跃的社区，提供了大量的第三方库和框架。",
            ],
        ],
    }

    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(f"\n[RAGAS] Result: {result}")

    for metric_name in ["faithfulness", "answer_relevancy"]:
        for score in result[metric_name]:
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"
    print("[RAGAS] All scores valid!")


def test_query_design_document():
    """Test by directly checking document content for expected answers."""
    test_cases = [
        ("发布年份", "1991"),
        ("设计哲学", "可读性"),
        ("包管理工具", "pip"),
    ]
    for keyword, expected in test_cases:
        found = any(expected in doc for doc in SAMPLE_DOCUMENTS)
        print(f"[Query Design] '{keyword}' → '{expected}': {'Found' if found else 'Missing'}")
        assert found, f"Expected '{expected}' not found in knowledge base"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])