import os
import google.generativeai as genai
import asyncio
import re
from dotenv import load_dotenv
import chainlit as cl
import requests

# Load environment variables from .env file (if used)
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it.")

genai.configure(api_key=GEMINI_API_KEY)

# Real web search via Tavily Search API
def web_search(query: str, count: int = 5) -> str:
    """Fetches real articles and blogs from Tavily Search API."""
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY environment variable not set. Please set it.")

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": count,
        "include_answer": False,
        "include_raw_content": False
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise ValueError(f"Tavily API error: {response.text}")
    results = response.json().get("results", [])
    if not results:
        return "- No results found."

    output = ""
    for item in results:
        title = item.get("title", "No Title")
        link = item.get("url", "#")
        description = item.get("content", "")
        output += f"- [{title}]({link})\n  - {description}\n"
    return output

# Input guardrail
def input_guardrail(query: str) -> str:
    if "agentic ai" not in query.lower():
        raise ValueError("Query must relate to Agentic AI trends.")
    return query

# Output guardrail
def output_guardrail(response: str) -> str:
    if re.search(r"\b(no data|future prediction)\b", response.lower()):
        raise ValueError("Response contains speculative or unverifiable claims.")
    return response

# Trend seeker logic
async def trend_seeker(query: str, model_name: str = "gemini-2.0-flash") -> str:
    query = input_guardrail(query)
    model = genai.GenerativeModel(model_name)

    raw_data = web_search(query)

    prompt = f"""
You are a "Trend Seeker" AI agent specializing in identifying and summarizing emerging trends in Agentic AI.

Below are real articles and blogs pulled from the web:
{raw_data}

Your job:
- Identify 3–5 emerging trends in Agentic AI from these sources.
- For each trend:
    - Give a title.
    - Summarize the trend.
    - Include at least one **clickable Markdown link** to a supporting blog/article.
    - Explain the business impact.
    - Avoid repetition.
- Write clear Markdown sections.
"""

    response = await model.generate_content_async(prompt)
    report = response.text

    report = output_guardrail(report)

    return report

# Chainlit integration with simulated streaming
@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()
    msg = cl.Message(content="")
    await msg.send()

    try:
        full_report = await trend_seeker(query)
        # Simulate streaming by splitting paragraphs
        chunks = full_report.split("\n\n")
        for chunk in chunks:
            await asyncio.sleep(0.3)
            await msg.stream_token(chunk + "\n\n")
    except Exception as e:
        await msg.stream_token(f"Error: {e}")
