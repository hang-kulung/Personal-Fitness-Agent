from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import  Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search, AgentTool, FunctionTool, load_memory
# from google.adk.apps.app import App, EventsCompactionConfig
from google.genai import types
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)
import asyncio
from dotenv import load_dotenv
import os
from datetime import datetime
load_dotenv()

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

search_agent = Agent(
    name="Web_Search_Specialist",
    model=Gemini(model="gemini-2.5-flash-lite",
                 retry_options=retry_config,
                 api_key=os.getenv("GOOGLE_API_KEY")),
    instruction="You are a specialist designed only to perform Google web searches and summarize the results. Do not answer questions without searching.",
    tools=[google_search] # ONLY google_search
)
search_agent_tool = AgentTool(search_agent)

def get_date()->dict:
    """Retuns current date in dict:
    current_date:YYYY-MM-DD format
    year: int
    month: int
    day: int (0=Monday, 6=Sunday)
    """
    return {
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "year": datetime.now().year,
        "month": datetime.now().month,  
        "day": datetime.now().weekday()
    }  

get_date_tool = FunctionTool(get_date)

trainer_agent = Agent(
    name="workout_trainer_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        retry_options=retry_config
    ),
    description="A simple trainer agent that can suggest work to do.",
    instruction="""Your only work is to act as a workout trainer agent.
    If user's info is missing, ask for it first.
    -> You can use "search_agent_tool" from tools to get access to web search if needed.
    -> Use user's preferences and user's description to create a well structure workout plan for 7-days.
    -> Must mention sets, reps, and rest times for each exercise.
    -> Consider available equipment and fitness level.
    -> Keep suggestions safe and effective. 
    -> Use "get_date_tool" from tools to get current date and then proivide a schedule for the workout for that day only from the premade 7-days plan.
    -> Stick to a fix plan. Do not change the plan unless user requests it.
    -> Manage plan according to user's feedback and progress over time. Use load_memory to get user's past data.
    -> Don't provide diet plan. Only provide workout plan.
    -> ONLY provide plan for other days when user asks for it specifically
    """,
    tools=[search_agent_tool, load_memory, get_date_tool],
    output_key="workout_plan",
)

diet_agent = Agent(
    name="health_diet_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        retry_options=retry_config
    ),
    description="A simple diet agent that can suggest foods to have.",
    instruction=""" Your only work is to act as a healthy diet agent.
    If user's info is missing, ask for it first.
    -> Use "search_agent_tool" from tools to get access to web search if needed.
    -> Suggest healthy diet based on user's discription.
    -> Consider the workout plan suggested by workout_trainer_agent.
    -> Mention diet for each time of the day i.e. Breakfast, Lunch, Snack and Dinner.
    -> Keep suggestions nutritious and balanced.
    -> Suggest alternatives for common dietary restrictions.
    -> Use "get_date_tool" from tools to get current date and then proivide a diet plan for that day only.
    -> Manage diet plan according to user's feedback and progress over time.
    -> Provide different diet plan than previous days.
    -> ONLY provide diet plan for other days when user asks for it specifically
    Use search_agent_tool for reference if needed.""",
    tools=[search_agent_tool, load_memory, get_date_tool],
    output_key="diet_plan",
)

async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

root_agent = Agent(
    name="Personal_Fitness_Agent",
    model=Gemini(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        retry_options=retry_config
    ),
    instruction="""You are a personal fitness agent.
    Initially get user's fitness goals, description(age, gender, weights) and preferences.
    -> First, use "workout_trainer_agent"  from AgentTool to create a workout plan for the day.
    -> Next, use "health_diet_agent"  from AgentTool to create a diet plan for the day considering the workout plan suggested by workout_trainer_agent.
    -> Finally, combine both plans into a comprehensive fitness plan and present it to the user.
    -> If user sending feedback about previous plans, use that to improve future plans.
    -> You can use "get_date_tool" froom tools to get current date and "search_agent_tool" from tools to use google searches as needed.
    """,
    tools=[AgentTool(trainer_agent), AgentTool(diet_agent), get_date_tool, search_agent_tool],
    after_agent_callback=auto_save_to_memory
)

db_url = "sqlite+aiosqlite:///agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)
memory_service = InMemoryMemoryService()

runner = Runner(agent=root_agent,
    app_name= "agents",
    plugins=[LoggingPlugin()],
    session_service=session_service,
    memory_service=memory_service
    )

async def chat_loop():
    print("Personal Fitness Agent (type 'exit' to stop)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = await runner.run_debug(user_input)
        # If using arun(), access like: response.output_key or response.output

if __name__ == "__main__":
    asyncio.run(chat_loop())