import os

import openmeteo_requests
from dotenv import load_dotenv
from mem0 import Memory
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


load_dotenv()  # Ensures .env variables are loaded
api_key = os.getenv("BASIC_AGENT_API_KEY")

# Set OpenAI API key for mem0 (it looks for OPENAI_API_KEY env var)
os.environ["OPENAI_API_KEY"] = api_key

# Initialize mem0 for memory management
memory = Memory()

provider = OpenAIProvider(api_key=api_key)
model = OpenAIChatModel("gpt-4o-mini", provider=provider)
agent = Agent[None, str](model=model, system_prompt="You are a helpful assistant.")


@agent.tool_plain
def get_weather_forecast(latitude: float, longitude: float) -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    openmeteo = openmeteo_requests.Client()
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m"],
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    current = response.Current()
    current_temperature_2m = current.Variables(0).Value()
    return current_temperature_2m


def main():
    print("Type 'quit' to exit.")

    # User identifier for mem0
    user_id = "default_user"

    # Initialize message_history for immediate conversation context (PydanticAI needs this)
    message_history = []

    user_message = input("> ")

    while user_message.lower() != "quit":
        # Retrieve relevant memories from mem0 before processing
        memories = memory.search(user_message, user_id=user_id, limit=5)

        # Build context from mem0 memories
        memory_context = ""
        if memories and "results" in memories and len(memories["results"]) > 0:
            memory_context = "Relevant memories:\n"
            for mem in memories["results"]:
                memory_context += f"- {mem['memory']}\n"
            memory_context += "\n"

        # Prepare the message with memory context if available
        enhanced_message = (
            f"{memory_context}{user_message}" if memory_context else user_message
        )

        # Run with conversation history
        result = agent.run_sync(enhanced_message, message_history=message_history)

        print(result.output)

        # Store conversation in mem0 for long-term memory
        messages_to_store = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": result.output},
        ]
        memory.add(messages_to_store, user_id=user_id)

        # Add the user message and response to message_history (for immediate context)
        message_history.append(
            ModelRequest(parts=[UserPromptPart(content=user_message)])
        )
        message_history.append(ModelResponse(parts=[TextPart(content=result.output)]))

        # Keep only the last 10 messages (5 user-response pairs) in immediate history
        if len(message_history) > 10:
            message_history = message_history[-10:]

        user_message = input("> ")


if __name__ == "__main__":
    main()
