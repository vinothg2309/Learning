"""
LLM Configuration Module
Creates and returns a configured ChatOpenAI instance for use across notebooks.

Usage in notebooks:
    from llm_config import get_llm

    llm = get_llm()
    response = llm.invoke("Your query here")
"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI


def get_llm(model="claude-haiku-4-5", temperature=0, max_tokens=8192):
    """
    Initialize and return a configured ChatOpenAI instance.

    Args:
        model (str): Model name to use. Default: "claude-sonnet-4.5"
                    Available models with vision capabilities:
                    - claude-sonnet-4.5 (recommended for vision tasks)
                    - claude-opus-4.5 (most capable, slower)
                    - gpt-4o (OpenAI vision model)
                    - gemini-2.5-pro (Google vision model)
        temperature (float): Sampling temperature. Default: 0
        max_tokens (int): Maximum tokens in response. Default: 4096
                         Increased for vision tasks and detailed summaries

    Returns:
        ChatOpenAI: Configured LLM instance

    Example:
        >>> llm = get_llm()
        >>> llm = get_llm(model="claude-opus-4.5", max_tokens=8192)
        >>> llm = get_llm(model="gpt-4o", max_tokens=2048)
    """
    # Load environment variables
    load_dotenv()

    # Get API credentials
    api_key = os.environ.get('UNIFIED_LLM_KEY')
    if not api_key:
        raise ValueError("UNIFIED_LLM_KEY not found in environment variables")

    # Base URL for the LLM service
    base_url = ""

    # Create and return LLM instance
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url
    )

    return llm


# For backward compatibility - create default instance
llm = get_llm()