"""Configuration for AI test project."""
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# Model configuration
MODEL_NAME = "deepseek-chat"
TEMPERATURE = 0.7
MAX_TOKENS = 1024

# Test configuration
DEFAULT_NUM_RUNS = 2  # runs per prompt for statistical significance
REPORT_DIR = "reports"