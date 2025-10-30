from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()  # Ensures .env variables are loaded
api_key = os.getenv("BASIC_AGENT_API_KEY")

provider = OpenAIProvider(api_key=api_key)
model = OpenAIChatModel("gpt-4o-mini", provider=provider)
agent = Agent[None, str](model=model, system_prompt="You are a helpful assistant.")


def main():
    print("Type 'quit' to exit.")
    user_message = input("> ")
    result = agent.run_sync(user_message)
    while user_message.lower() != "quit":
        print(result.output)
        user_message = input("> ")
        if user_message.lower() != "quit":
            # Run with conversation history
            result = agent.run_sync(user_message, message_history=result.all_messages())


if __name__ == "__main__":
    main()
