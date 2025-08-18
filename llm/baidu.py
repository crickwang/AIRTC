from openai import OpenAI
import time

class BaiduClient:
    """
    Baidu LLM client for interacting with the Baidu API.
    """
    def __init__(
        self: object, 
        model: str, 
        is_local: bool = False, 
        api_key: str = None, 
        base_url: str = None
    ) -> None:
        '''
        Initialize the Baidu LLM client with system prompt and user/assistant messages.
        Args:
            model (str): The model to be used for the LLM.
            is_local (bool, optional): If True, use a local model instead of the API. 
                Defaults to False.
            api_key (str, optional): API key for accessing the Baidu LLM API.
            base_url (str, optional): Base URL for the Baidu LLM API.
        Note:
            When is_local is True, api_key and base_url are not required.
        Raises:
            ValueError: If api_key or base_url is None when is_local is False.
        '''
        self.model = model
        self.api_key = api_key  
        self.base_url = base_url
        self.client = None
        if not is_local:
            self.connect()

    def change_model(self: object, model: str) -> None:
        '''
        Change the model used by the LLM client.
        Args:
            model (str): The new model to be used.
        '''
        self.model = model

    def connect(self: object) -> None:
        '''
        Connect to the Baidu LLM API using the provided API key and base URL.
        Raises:
            ValueError: If api_key or base_url is None.
        '''
        try:
            if self.api_key is None or self.base_url is None:
                raise ValueError("API key and base URL must be provided.")
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            print(f"Error in connecting to API: {e}")

    def message(
        self: object, 
        users: list = None, 
        system: str = None, 
        assistants: list = None
        ) -> list:
        """
        Construct a list of messages for the Baidu LLM API.
        Args:
            users (list, optional): List of user messages. Defaults to None.
            system (str, optional): System prompt for the LLM. Defaults to None.
            assistants (list, optional): List of assistant messages. Defaults to None.
        Note:
            1. Users list must contain at least one user message.
            2. Number of users must be at least 1.
            3. Number of assistants must be equal to number of users - 1.
            4. If no assistants are provided, only the first user message is used.
        Returns:
            list: A list of messages formatted for the Baidu LLM API.
        """
        messages = []
        if users is None or len(users) == 0:
            raise ValueError("Users list cannot be empty.")

        if system is not None and len(system) > 0:
            messages.append({'role': 'system', 'content': system})

        if assistants is None or len(assistants) == 0:
            if users is None or len(users) != 1:
                raise ValueError("Please provide a single user message if "
                                 "no assistant messages are provided.")
            messages.append({'role': 'user', 'content': users[0]})
        else:
            if (len(users) - 1) != len(assistants):
                raise ValueError("The number of user messages must match the "
                                 "number of assistant messages + 1.")
            messages.append({'role': 'user', 'content': users[0]})
            for user, assistant in zip(users[1:], assistants):
                messages.append({'role': 'assistant', 'content': assistant})
                messages.append({'role': 'user', 'content': user})
        return messages

    def generate(
        self: object, 
        users: list = None, 
        system: str = None, 
        assistants: list = None, 
        max_tokens: int = 512
        ) -> str:
        '''
        Generate a response from the Baidu LLM using the provided API key and base URL.
        Args:
            users (list, optional): List of user messages. Defaults to None.
            system (str, optional): System prompt for the LLM. Defaults to None.
            assistants (list, optional): List of assistant messages. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens for the response. Defaults to 512.
        Note:
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
        start_time = time.time()
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            max_completion_tokens=max_tokens,
        )
        print(f"Request sent at {time.time() - start_time}, waiting for response...")
        return chat_completion

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("BAIDU_AISTUDIO_API_KEY")
    base_url = os.environ.get("BAIDU_AISTUDIO_BASE_URL")

    model = "ernie-4.5-0.3b"
    system_prompt = "You are a helpful assistant."
    users = ["count from 1 to 10 and 10 to 0 plz? Be more elaborate."]
    buffer = ''
    baidu_client = BaiduClient(model=model, api_key=api_key, base_url=base_url)
    chat_completion = baidu_client.generate(users=users, system=system_prompt, max_tokens=32)
    output = []
    for chunk in chat_completion:
        response = chunk.choices[0].delta.content
        print(response, end='', flush=True)
        prev = 0
        for i, char in enumerate(response):
            if char != '\n':
                if char in ['。', '！', '？', '.', '!', '?']:
                    # If punctuation, send the buffer
                    if buffer:
                        temp_text = buffer + response[prev:i+1]
                        temp_text = temp_text.strip(' \n*\"')
                        output.append(temp_text)
                        buffer = ""
                    else:
                        temp_text = response[prev:i+1]
                        temp_text = temp_text.strip(' \n*\"')
                        output.append(temp_text)
                    prev = i + 1
            else:
                prev = i + 1
        buffer += response[prev:]
        buffer.strip(' \n*\"')
    output.append(buffer.strip(' \n*\"'))
    print()
    print(output)