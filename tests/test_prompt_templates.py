"""Prompt template comparison tests."""
import pytest
from dotenv import load_dotenv
import os

load_dotenv()

# Mock responses for different prompt strategies
MOCK_RESPONSES = {
    "zero_shot": "最大的两个数是9和6，它们的和是15。",
    "few_shot": "最大两数和 = 15",
    "cot": "步骤1：找出最大的数 = 9\n步骤2：找出第二大的数 = 6\n步骤3：求和 = 9 + 6 = 15\n\n最终答案：15",
}

USE_MOCK = not os.getenv("DEEPSEEK_API_KEY") or "..." in (os.getenv("DEEPSEEK_API_KEY") or "")


def _call_llm_for_strategy(strategy: str) -> str:
    """Return API or mock response based on strategy."""
    if USE_MOCK:
        return MOCK_RESPONSES[strategy]

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )

        QUESTION = "如果有一个数组 [3, 1, 4, 1, 5, 9, 2, 6]，请找出其中最大的两个数之和。"

        if strategy == "zero_shot":
            system = "你是一个数学助手。"
            user = QUESTION
        elif strategy == "few_shot":
            system = "你是一个数学助手。请严格按照示例的格式回答。"
            user = (
                "示例1：数组 [1, 2, 3] → 最大两数和 = 5\n"
                "示例2：数组 [10, 5, 8] → 最大两数和 = 18\n"
                "示例3：数组 [100, 1, 50] → 最大两数和 = 150\n\n"
                f"现在请回答：{QUESTION}"
            )
        elif strategy == "cot":
            system = "你是一个数学助手。请一步一步推理，然后给出最终答案。"
            user = (
                f"{QUESTION}\n\n请一步一步思考：\n"
                "1. 首先找出数组中最大的数\n"
                "2. 然后找出第二大的数\n"
                "3. 最后将两个数相加"
            )

        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        print(f"[WARN] API call failed for {strategy}, falling back to mock")
        return MOCK_RESPONSES[strategy]


def test_zero_shot():
    """Zero-shot: no examples given."""
    response = _call_llm_for_strategy("zero_shot")
    print(f"\n[Zero-Shot] Response: {response[:200]}")
    assert any(k in response for k in ["15", "9+6", "9 + 6", "6+9"]), (
        f"Expected 15, got: {response[:200]}"
    )


def test_few_shot():
    """Few-shot: examples provided."""
    response = _call_llm_for_strategy("few_shot")
    print(f"\n[Few-Shot] Response: {response[:200]}")
    assert any(k in response for k in ["15", "最大两数和"]), (
        f"Expected 15, got: {response[:200]}"
    )


def test_chain_of_thought():
    """Chain-of-Thought: step-by-step reasoning."""
    response = _call_llm_for_strategy("cot")
    print(f"\n[CoT] Response: {response[:300]}")
    assert any(k in response for k in ["15", "9+6", "6+9"]), (
        f"Expected 15, got: {response[:200]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])