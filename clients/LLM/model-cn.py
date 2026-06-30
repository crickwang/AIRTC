# 该文档的注释由AI生成，旨在帮助开发者理解代码结构和功能。
# 具体执行请使用英文版 （无-cn后缀）

class LLMClient(ABC):
    def __init__(self, model: str, api_key: str, base_url: str, **kwargs):
        """
        初始化 LLM 客户端。
        参数:
            model: 要使用的模型。
            api_key: LLM 的 API 密钥。
            base_url: LLM API 的基础 URL。
        """

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
        从 LLM 生成响应。
        工作流程：
        1. 从 input_queue 接收转录结果。
        2. 调用 LLM 生成响应。
        3. 将响应放入 output_queue。
        4. stop_event 用于终止程序。
        5. interrupt_event 用于打断其他客户端。
        """



class BaiduLLM(LLMClient):
    def __init__(self, model: str, api_key: str, base_url: str, **kwargs):
        """
        初始化 Baidu LLM 客户端。
        参数:
            model: 使用的模型。
            api_key: Baidu LLM API key。
            base_url: Baidu LLM API 基础 URL。
        """

    def change_model(self: object, model: str) -> None:
        """
        更换当前 LLM 使用的模型。
        参数:
            model: 新模型。
        """

    def connect(self: object) -> None:
        """
        使用提供的 API key 和 base_url 连接 Baidu LLM API。
        """

    def message(self: object, users: list = None, system: str = None, assistants: list = None) -> list:
        """
        构建 LLM 输入消息列表。
        参数:
            users: 用户输入的消息列表。
            system: 系统提示。
            assistants: 助手回复列表。
        返回:
            格式化后的消息列表。
        说明:
            - users 必须至少包含一个用户消息。
            - assistants 数量 = users 数量 - 1。
        """

    def chat(
        self: object,
        users: list = None,
        system: str = None,
        assistants: list = None,
        max_tokens: int = 512
    ) -> str:
        """
        使用 Baidu LLM 生成回复。
        参数:
            users: 用户消息。
            system: 系统提示。
            assistants: 助手回复。
            max_tokens: 最大 token 数，默认 512。
        返回:
            LLM 的流式回复。
        """

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
        基于用户转录生成文本响应。
        参数:
            input_queue: ASR 输入结果队列。
            output_queue: 发送给 TTS 的回复队列。
            stop_event: 停止信号。
            interrupt_event: 打断信号。
            system: 系统提示。
            max_tokens: 最大 token 数。
            timeout: 超时。
        """


class LLMClientFactory:
    @staticmethod
    def create(platform, **kwargs) -> LLMClient:
        """
        创建一个 LLM 客户端实例。
        参数:
            platform (str): 要创建的 LLM 客户端所属平台名称。
        返回:
            LLMClient: 创建好的 LLM 客户端实例。
        """
