from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
import openmeteo_requests
import pprint

load_dotenv()  # Ensures .env variables are loaded
api_key = os.getenv("BASIC_AGENT_API_KEY")

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

    # Initialize message_history as an empty list
    message_history = []

    user_message = input("> ")

    while user_message.lower() != "quit":
        # Run with conversation history
        result = agent.run_sync(user_message, message_history=message_history)

        print(result.output)
        """ print("--------------------------------")
        print(result.all_messages())
        print("--------------------------------") """

        # Add the user message and response to message_history (only text content, no metadata)
        message_history.append(
            ModelRequest(parts=[UserPromptPart(content=user_message)])
        )
        message_history.append(ModelResponse(parts=[TextPart(content=result.output)]))

        # Keep only the last 10 messages (5 user-response pairs)
        if len(message_history) > 10:
            message_history = message_history[-10:]

        user_message = input("> ")


if __name__ == "__main__":
    main()
