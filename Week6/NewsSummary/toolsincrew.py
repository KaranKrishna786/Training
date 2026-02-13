import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from crewai_tools import SerpApiGoogleSearchTool
from dotenv import load_dotenv


load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise EnvironmentError("GEMINI_API_KEY not set")
 
# Pick a supported model your account has access to:
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",     # or "gemini/gemini-2.0-flash"
    api_key=gemini_key,
    temperature=0.2,
)

# Define a custom tool
@tool("search_google")
def google_search(query: str) -> str:
    """Searches Google and returns a simulated answer."""
    # In a real scenario, this would integrate with an actual search API
    return f"Simulated search result for: {query}"

# Add the tool to an agent
researcher = Agent(
    role="AI Researcher",
    goal="Find the latest trends in generative AI.",
    backstory="You use external tools to get real-time information.",
    tools=[google_search],
    verbose=True,
    llm = gemini_llm
)

# The rest of the setup (Task, Crew) remains similar

# Define Tools
search_tool = SerpApiGoogleSearchTool() # Requires a SERPAPI_API_KEY environment variable

# Define Agents
researcher = Agent(
    role="AI News Researcher",
    goal="Find 5 important AI news stories from the last 7 days",
    backstory="You are an AI news expert, skilled in finding trending topics.",
    tools=[search_tool],
    verbose=True,
    llm = gemini_llm
)

writer = Agent(
    role="Content Creator",
    goal="Write an engaging LinkedIn post from an AI news summary",
    backstory="You are an expert in writing viral LinkedIn content for engineers.",
    verbose=True,
    llm = gemini_llm
)

# Define Tasks
research_task = Task(
    description="Search for the 5 most interesting AI news stories from the last 7 days.",
    expected_output="A list of 5 headlines and a 1-line description for each.",
    agent=researcher
)

write_post_task = Task(
    description="Using the news items, write a LinkedIn-style post aimed at engineering leaders.",
    expected_output="A 200-word professional and engaging post.",
    agent=writer,
    context=[research_task]
)

# Create and run the Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_post_task],
    verbose=True
)

result = crew.kickoff()
print("\nFinal LinkedIn Post:\n")
print(result)
