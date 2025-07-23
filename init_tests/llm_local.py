from ollama import chat
import yaml

with open('llm_test.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

system = config.get('system')
users = config.get('user')
assistants = config.get('assistant')
model = config.get('local_model', 'deepseek-r1:1.5b')

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

stream = chat(
    model=model,
    messages=messages,
    stream=True,
)

for chunk in stream:
    content = chunk['message']['content']
    print(content, end='', flush=True)
