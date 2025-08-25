from abc import ABC, abstractmethod
from openai import OpenAI
import time
import asyncio
import traceback
import unicodedata as ucd
from register import register
from config.constants import TIMEOUT
from clients.utils import log_to_client

class LLMClient(ABC):
    def __init__(self, model: str, api_key: str, base_url: str, **kwargs):
        """
        Initialize the LLM client with the specified model, API key, and base URL.
        Args:
            model (str): The model to be used for the LLM.
            api_key (str): The API key for accessing the LLM.
            base_url (str): The base URL for the LLM API.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        
    @abstractmethod
    async def generate(
        self: object, 
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        **kwargs
    ):
        """
        Generate a response from the LLM.
        Args:
            input_queue (asyncio.Queue): The input queue containing user messages.
            output_queue (asyncio.Queue): The output queue for sending responses.
            stop_event (asyncio.Event): The event to signal stopping the generation.
            interrupt_event (asyncio.Event): The event to signal interrupting the generation.
            **kwargs: Additional keyword arguments.
        """
        pass

@register.add_model("llm", "baidu")
class BaiduLLM(LLMClient):
    def __init__(
        self, 
        model: str, 
        api_key: str, 
        base_url: str, 
        **kwargs,
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
        super().__init__(model, api_key, base_url)
        self.connect()
        self.pc = kwargs.get('pc', None)

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
            msg = f"LLM: Error connecting to Baidu LLM API: {e}"
            print(msg)
            if self.pc:
                log_to_client(self.pc.log_channel, msg)

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

    def chat(
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
        Returns:
            str: The generated streaming response from the Baidu LLM.
        '''
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
        return chat_completion
    
    async def generate(
        self: object,
        input_queue: asyncio.Queue, 
        output_queue: asyncio.Queue, 
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        system: str = None, 
        max_tokens: int = 512,
        timeout: int = TIMEOUT,
        ) -> None:
        """
        LLM that generates texts based on the incoming trancribed message of the user
        Args:
            input_queue (asyncio.Queue): The queue to receive user input from ASR.
            output_queue (asyncio.Queue): The queue to send the generated response to TTS.
            stop_event (asyncio.Event): Event to signal when to stop the program.
            interrupt_event (asyncio.Event): Event to signal when to interrupt the current LLM generation.
            system (str, optional): The system prompt to use for the LLM, defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate, defaults to 512.
            timeout (float, optional): The timeout for the LLM request, defaults to 600 seconds.
        Returns:
            None
        Raises:
            Exception: If an error occurs during text generation.
        """
        try:
            texts = []
            assistants = []
            while not stop_event.is_set():
                text = await asyncio.wait_for(input_queue.get(), timeout=timeout)
                texts.append(text)
                if stop_event.is_set() or text is None:
                    await output_queue.put(None)
                    return 
                print(f"LLM: Generating response for: '{text}'")
                chat_completion = self.chat(
                    users=texts,
                    system=system,
                    assistants=assistants,
                    max_tokens=max_tokens,
                )
                total_response = ""
                # Buffer for combining small chunks
                buffer = ""
                for chunk in chat_completion:
                    response = chunk.choices[0].delta.content
                    total_response += response
                    prev = 0
                    # split each response based on common puntuations.
                    # Strip the uncommon symbols for each chunk.
                    temp_text = ""
                    for i in range(len(response)):
                        char = response[i]
                        if stop_event.is_set():
                            await output_queue.put(None)
                            return
                        if interrupt_event.is_set():
                            await output_queue.put(None)
                            break
                        # skip any other characters
                        if ucd.category(char).startswith('P') or ucd.category(char).startswith('S'):
                            if char in ['。', '，', '！', '？', '.', ',', '!', '?']:
                                # If punctuation, send the buffer
                                if i < prev + 10:
                                    response = response[:i] + ' ' + response[i+1:]
                                    continue
                                if buffer:
                                    temp_text = buffer + response[prev:i]
                                    buffer = ""
                                else:
                                    temp_text = response[prev:i]
                                msg = f"LLM: Sending text into TTS: '{temp_text}'"
                                print(msg)
                                if self.pc:
                                    log_to_client(self.pc.log_channel, msg)
                                await output_queue.put(temp_text)
                                prev = i + 1
                            else:
                                response = response[:i] + ' ' + response[i+1:]
                    buffer += response[prev:]
                    if interrupt_event.is_set():
                        print("LLM: Interrupt event set, stopping generation")
                        await output_queue.put(None)
                        break

                if interrupt_event.is_set():
                    print("LLM: Interrupt event set, stopping generation")
                    await output_queue.put(None)
                elif buffer:
                    await output_queue.put(buffer)
                assistants.append(total_response)

        except Exception as e:
            msg = f"LLM: Error: {e}"
            print(msg)
            if self.pc:
                log_to_client(self.pc.log_channel, msg)
            traceback.print_exc()
        finally:
            # Always send end marker
            await output_queue.put(None)
            msg = "LLM: end of processing"
            print(msg)
            if self.pc:
                log_to_client(self.pc.log_channel, msg)

class LLMClientFactory:
    @staticmethod
    def create(platform, **kwargs) -> LLMClient:
        """
        Create a LLM client instance.
        Args:
            platform (str): The platform name to create the LLM client for.
        Returns:
            LLMClient: The created LLM client.
        """
        try:
            client = register.get_model("llm", platform)
            return client(**kwargs)
        except Exception as e:
            print(f"Error creating LLM client: {e}")
            return None
            
