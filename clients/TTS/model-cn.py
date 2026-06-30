# 该文档的注释由AI生成，旨在帮助开发者理解代码结构和功能。
# 具体执行请使用英文版 （无-cn后缀）

class TTSClient(ABC):
    def __init__(self, **kwargs):
        """
        抽象 TTS 客户端基类。
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
        使用 TTS 服务将文本转为语音。
        工作流程：
        1. 从 input_queue 接收 LLM 生成的文本。
        2. 处理文本并生成语音。
        3. 将语音放入 output_queue。
        4. stop_event 用于终止。
        5. interrupt_event 用于打断生成。
        """


@register.add_model("tts", "azure")
class AzureTTS(TTSClient):
    def __init__(self, voice: str, key: str, region: str, **kwargs):
        """
        初始化 Azure TTS 实例。
        参数:
            voice: 要使用的语音名称。
            key: Azure 认知服务的 API key。
            region: Azure 服务区域。
        """

    async def generate(
        self: object,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        samples_per_frame: int,
        timeout: int = TIMEOUT
    ) -> None:
        """
        Azure TTS 生成。
        参数:
            input_queue: LLM 输出的文本队列。
            output_queue: 输出音频队列。
            stop_event: 停止信号。
            interrupt_event: 打断信号。
            samples_per_frame: 每帧音频样本数。
            timeout: 超时时间。
        """

    async def process_text(
        self: object,
        sentence: str,
        output_queue: asyncio.Queue,
        samples_per_frame: int,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        sentence_id: int,
    ) -> int:
        """
        处理单个句子的 TTS 转换。
        使用 Azure 真·流式接口，边生成边输出音频。
        """

    async def send_audio(
        self: object,
        audio_np: np.ndarray,
        output_queue: asyncio.Queue,
        samples_per_frame: int
    ) -> None:
        """
        立即发送音频块，而不是等待整句结束。
        """


@register.add_model("tts", "edge")
class EdgeTTS(TTSClient):
    def __init__(self, voice: str, output_file: str = None, srt_file: str = None):
        """
        初始化 Edge TTS 实例。
        参数:
            voice: 语音名称。
            output_file: 可选，保存输出音频文件。
            srt_file: 可选，保存字幕文件。
        """

    async def generate(
        self: object,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        samples_per_frame: int,
        timeout: int = TIMEOUT
    ) -> None:
        """
        Edge TTS 生成。
        参数:
            input_queue: 输入文本队列。
            output_queue: 输出音频队列。
            stop_event: 停止信号。
            interrupt_event: 打断信号。
            samples_per_frame: 每帧音频样本数。
            timeout: 超时。
        """

    async def process_text(
        self: object,
        sentence: str,
        output_queue: asyncio.Queue,
        samples_per_frame: int,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        sentence_id: int,
    ) -> int:
        """
        处理单个句子的 TTS 转换。
        边生成边输出音频。
        """

    async def send_audio(
        self: object,
        audio_np: np.ndarray,
        output_queue: asyncio.Queue,
        samples_per_frame: int
    ) -> None:
        """
        立即发送音频块，而不是等待整句结束。
        """


@register.add_model("tts", "google")
class GoogleTTS(TTSClient):
    def __init__(self, voice: str, language: str):
        """
        初始化 Google TTS 实例。
        参数:
            voice: 语音名称。
            language: 语言编码。
        """

    async def generate(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        timeout=TIMEOUT,
    ) -> None:
        """
        Google TTS 生成。
        参数:
            input_queue: 输入文本队列。
            output_queue: 输出音频队列。
            stop_event: 停止信号。
            interrupt_event: 打断信号。
            timeout: 超时。
        """

class TTSClientFactory:
    @staticmethod
    def create(platform, **kwargs) -> TTSClient:
        """
        创建一个 TTS 客户端实例。
        参数:
            platform (str): 要创建的 TTS 客户端所属平台名称。
        返回:
            TTSClient: 创建好的 TTS 客户端实例。
        """
