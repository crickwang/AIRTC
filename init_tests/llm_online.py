import os
from openai import OpenAI
from dotenv import load_dotenv
import yaml

with open('llm_test.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
system = config.get('system')
users = config.get('user')
assistants = config.get('assistant')

load_dotenv()
client = OpenAI(
     api_key=os.environ.get("API_KEY"),
     base_url=os.environ.get("BASE_URL"),
)

messages = []
if not system is None:
    messages.append({'role': 'system', 'content': system})
    
if assistants is None:
    if len(users) != 1:
        raise ValueError("Please provide a single user message if no assistant messages are provided.")
    messages.append({'role': 'user', 'content': users[0]})
else:
    if (len(users) - 1) != len(assistants):
        raise ValueError("The number of user messages must match the number of assistant messages + 1.")
    messages.append({'role': 'user', 'content': users[0]})
    for user, assistant in zip(users[1:], assistants):
        messages.append({'role': 'assistant', 'content': assistant})
        messages.append({'role': 'user', 'content': user})

chat_completion = client.chat.completions.create(
    messages=messages,
    model=os.environ.get("MODEL"),
    stream=True,
    max_completion_tokens = 512,
)

for chunk in chat_completion:
    print(chunk.choices[0].delta.content or "", end="")