import dotenv
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_cohere import ChatCohere

from dotenv import load_dotenv
load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Returns weather for a city (dummy function)."""
    return f"The weather in {city} is 30°C and sunny."

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", google_api_key="AIzaSyCN0Esg5nooULYxSO7EO82RTmacXnwjzx0")
llm = ChatCohere(model="command-a-03-2025", temperature=0)

def create_weather_agent():
    # 1. Simplify the prompt. 
    # LangGraph automatically handles the user input from the message history.
    # You only need to provide the instructions (System Prompt).
    instructions = "You are a weather assistant. Use tools to answer questions."

    tools = [get_weather]

    # 2. Pass the instructions directly to the 'state_modifier' or 'prompt' parameter.
    # In newer versions, 'prompt' is the preferred argument name.
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=instructions, # This becomes the System Message automatically
        name="weather_agent"
    )
    return agent

from langchain_core.messages import HumanMessage

def main():
    # 1. Initialize the agent
    weather_agent = create_weather_agent()

    print("--- Weather Agent initialized (Cohere Command-A) ---")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # 2. Prepare the input for the agent
        # LangGraph ReAct agents expect a state dictionary with a 'messages' key
        inputs = {"messages": [HumanMessage(content=user_input)]}

        # 3. Execute the agent
        # We use stream() to see the Thought/Action process in real-time
        print("\nAgent is thinking...")
        
        try:
            # stream_mode="values" will yield the full state after each node execution
            for event in weather_agent.stream(inputs, stream_mode="values"):
                # Get the last message in the list (the most recent update)
                last_message = event["messages"][-1]
                
                # pretty_print() is a built-in LangChain method to show content clearly
                last_message.pretty_print()
                
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

