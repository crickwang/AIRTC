from openai import OpenAI

class BaiduClient:
    def __init__(self, model, users, system=None, assistants=None, max_tokens=512):
        '''
        Initialize the Baidu LLM client with system prompt and user/assistant messages.
        Args:
            system (str): System prompt for the LLM.
            users (list): List of user messages.
            assistants (list): List of assistant messages.
        '''
        self.system = system
        self.users = users
        self.assistants = assistants
        self.model = model
        self.max_tokens = max_tokens

    def message(self):
        messages = []
        if self.users is None or len(self.users) == 0:
            raise ValueError("Users list cannot be empty.")
        
        if not self.system is None:
            messages.append({'role': 'system', 'content': self.system})

        if self.assistants is None:
            if len(self.users) != 1:
                raise ValueError("Please provide a single user message if no assistant messages are provided.")
            messages.append({'role': 'user', 'content': self.users[0]})
        else:
            if (len(self.users) - 1) != len(self.assistants):
                raise ValueError("The number of user messages must match the number of assistant messages + 1.")
            messages.append({'role': 'user', 'content': self.users[0]})
            for user, assistant in zip(self.users[1:], self.assistants):
                messages.append({'role': 'assistant', 'content': assistant})
                messages.append({'role': 'user', 'content': user})
        return messages
    
    def generate(self, api_key, base_url):
        '''
        Generate a response from the Baidu LLM using the provided API key and base URL.
        Args:
            api_key (str): API key for authentication.
            base_url (str): Base URL for the Baidu LLM API.
            
        Reference:
            https://ai.baidu.com/ai-doc/AISTUDIO
        '''
        res = ''
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            messages = self.message()
        except ValueError as e:
            print(f"Error in message construction: {e}")
            return

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            max_completion_tokens=self.max_tokens,
        )

        for chunk in chat_completion:
            res += chunk.choices[0].delta.content or ""
        return res


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")

    model = "ernie-4.5-0.3b"
    system_prompt = "You are a helpful assistant."
    users = ["count from 1 to 10 and 10 to 0 plz?"]
    
    baidu_client = BaiduClient(model=model, users=users, system=system_prompt)
    response = baidu_client.generate(api_key=api_key, base_url=base_url)
    print(response)