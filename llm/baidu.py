from openai import OpenAI

class BaiduClient:
    def __init__(self, model, is_local=False, api_key=None, base_url=None):
        '''
        Initialize the Baidu LLM client with system prompt and user/assistant messages.
        Args:
            model (str): The model to be used for the LLM.
            is_local (bool): If True, use a local model instead of the API.
            api_key (str): API key for accessing the Baidu LLM API.
            base_url (str): Base URL for the Baidu LLM API.
        '''
        self.model = model
        self.api_key = api_key  
        self.base_url = base_url
        
        if is_local:
            self.client = None  # Replace with local client, TODO
        else:
            self.connect()

    def change_model(self, model):
        '''
        Change the model used by the LLM client.
        Args:
            model (str): The new model to be used.
        '''
        self.model = model
    
    def connect(self):
        '''
        Connect to the Baidu LLM API using the provided API key and base URL.
        '''
        try:
            if self.api_key is None or self.base_url is None:
                raise ValueError("API key and base URL must be provided.")
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            print(f"Error in connecting to API: {e}")

    def message(self, users=None, system=None, assistants=None):
        messages = []
        if users is None or len(users) == 0:
            raise ValueError("Users list cannot be empty.")

        if system is not None:
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
        return messages

    def generate(self, users=None, system=None, assistants=None, max_tokens=512):
        '''
        Generate a response from the Baidu LLM using the provided API key and base URL.
        Args:
            users (list): List of user messages.
            system (str): System prompt for the LLM.
            assistants (list): List of assistant messages.
            max_tokens (int): Maximum number of tokens for the response.
                1. Number of users must be at least 1.
                2. Number of assistants must be equal to number of users - 1.

        Reference:
            https://ai.baidu.com/ai-doc/AISTUDIO
        '''
        res = ''
        try:
            messages = self.message(users=users, system=system, assistants=assistants)
        except ValueError as e:
            print(f"Error in constructing messages: {e}")
            return

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            max_completion_tokens=max_tokens,
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
    
    baidu_client = BaiduClient(model=model, api_key=api_key, base_url=base_url)
    response = baidu_client.generate(users=users, system=system_prompt)
    print(response)