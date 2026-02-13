import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_core.models import ModelInfo
from dotenv import load_dotenv
import os

async def main() -> None:
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in your .env")

    # Gemini via OpenAI-compatible endpoint (official method)
    model_client = OpenAIChatCompletionClient(
        model="gemini-3-flash-preview",
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_info=ModelInfo(
            vision=True, function_calling=True, json_output=True,
            structured_output=True, family="unknown",
        ),
        max_retries=2,   # a couple of automatic retries on transient failures
        timeout=60,
    )

    # WebSurfer tuned to avoid Google CAPTCHA
    web_surfer_agent = MultimodalWebSurfer(
        name="MultimodalWebSurfer",
        model_client=model_client,
        # Key tweaks:
        headless=True,                               # set False while debugging
        start_page="https://www.bing.com/",          # avoid starting on Google
        browser_data_dir=str(os.path.abspath("./.browser_profile")),  # persist cookies
        to_save_screenshots=False,
    )

    # Light behavioral guidance to the agent
    system_hint = (
        "When you need to search, use the page's search box once, then open 1–2 "
        "relevant results. Avoid rapid repeated queries. Wait ~2 seconds between actions."
        "Give me Phone number of the person I am searching for"
    )

    agent_team = RoundRobinGroupChat([web_surfer_agent], max_turns=3)
    stream = agent_team.run_stream(
        task=f"{system_hint}\nNavigate to Bing and search for 'Karan Krishna', then open the most relevant result."
    )
    await Console(stream)

    await web_surfer_agent.close()

if __name__ == "__main__":
    asyncio.run(main())