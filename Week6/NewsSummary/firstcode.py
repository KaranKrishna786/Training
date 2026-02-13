import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
 
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
 
# researcher = Agent(
#     role="AI Researcher",
#     goal="Find credible, recent AI trends and summarize them with citations.",
#     backstory="You analyze reputable sources and write concise summaries.",
#     llm=gemini_llm,                    # important: bind Gemini, avoid OpenAI default
#     verbose=True,
# )
 
# research_task = Task(
#     description="Identify the top 3 AI trends in 2024 using reputable online sources.",
#     expected_output=(
#         "A numbered list of the top 3 trends in 2024. For each: "
#         "• Trend name • 2–3 sentence summary • 1–2 source links"
#     ),
#     agent=researcher,
#     verbose=True,
# )
 
# crew = Crew(agents=[researcher], tasks=[research_task], verbose=True)
 
# if __name__ == "__main__":
#     # Newer CrewAI uses kickoff()/kickoff_async()
#     result = crew.kickoff(inputs={"format_hint": "Use markdown bullets and clickable links."})
#     print(result)
 
 
# Define Agents
researcher = Agent(
    role="AI Researcher",
    goal="Find recent breakthroughs in AI.",
    backstory="An expert in AI keeping up with the latest research.",
    verbose=True,
    llm=gemini_llm,
)

writer = Agent(
    role="Technical Writer",
    goal="Write a short blog post from research data.",
    backstory="A skilled writer who can turn complex info into engaging posts.",
    verbose=True,
    llm=gemini_llm,
)

# Define Tasks with context
research_task = Task(
    description="Find the 3 most important AI research breakthroughs of 2024.",
    expected_output="A list of 3 breakthroughs with a 1-2 line explanation each.",
    agent=researcher
)

write_task = Task(
    description="Write a short blog post based on the AI research findings.",
    expected_output="A 300-word blog post summarizing the breakthroughs.",
    agent=writer,
    context=[research_task]  # The writer uses the output of the research_task
)

# Define and run the Crew
ai_blog_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True
)

crew = Crew(agents=[researcher], tasks=[research_task], verbose=True)

if __name__ == "__main__":
    result = ai_blog_crew.kickoff()
    print(result)