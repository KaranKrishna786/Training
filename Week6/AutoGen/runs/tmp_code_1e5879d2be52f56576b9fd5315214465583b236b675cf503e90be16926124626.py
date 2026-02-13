import os
import autogen

# 1. LLM Configuration
# You can also use "gpt-3.5-turbo" for lower cost.
config_list = [
    {
        "model": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
]

llm_config = {
    "config_list": config_list,
    "cache_seed": 42, # Caching helps save credits during testing
}

# 2. Define the Assistant Agent
# This agent acts as the programmer/coder.
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="""You are a helpful AI assistant. 
    Solve tasks using your coding and language skills.
    In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file.
    2. When you need to perform some task with code, use the code to perform the task and output the result. 
    Check the execution result returned by the user.
    If the result indicates there is an error, fix the error and output the code again.
    Suggest the full code in a single block.
    When the task is done, reply with TERMINATE."""
)

# 3. Define the User Proxy Agent
# This agent executes the code provided by the assistant.
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER", # Set to NEVER to run automatically
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding_workspace", # Files will be created in this directory
        "use_docker": False,            # Set to True if you want execution in a Docker container
    },
)

# 4. Initiate the task
user_proxy.initiate_chat(
    assistant,
    message="""
    1. Create a JavaScript file named 'hello.js' that prints 'Hello World' to the console.
    2. Run the file using Node.js and show the output.
    """
)
