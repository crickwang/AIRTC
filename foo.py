import yaml
with open('llm_test.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
text = config.get('assistant')
print(len('  '.strip()))