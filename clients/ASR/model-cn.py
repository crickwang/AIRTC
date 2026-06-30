# 该文档的注释由AI生成，旨在帮助开发者理解代码结构和功能。
# 具体执行请使用英文版 （无-cn后缀）

class ASRClient(ABC):
    def __init__(self, **kwargs):
        """
        抽象 ASR 客户端基类。
        """

    @abstractmethod
    async def generate(
        self: object,
        track: AudioStreamTrack,
        output_queue: asyncio.Queue,
        audio_player: AudioStreamTrack,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        **kwargs,
    ):
        """
        使用 ASR 从音频流生成转录。
        工作流程：
        1. 从 WebRTC track 接收音频帧。
        2. 处理音频帧，生成转录，并放入 output_queue，传递给下一个 LLM。
        3. 使用 audio_player 向前端发出中断信号。
        4. 使用 stop_event 信号来停止整个程序。
        5. 使用 interrupt_event 信号来中断其他后端任务。
        """



class GoogleASR(ASRClient):
    def __init__(self, rate: int, language_code: str, chunk_size: int, stop_word: str = None, **kwargs):
        """
        初始化 Google ASR 客户端。
        参数:
            rate: 采样率，默认 24000。
            language_code: 语言编码，默认 'zh-CN'。
            chunk_size: 音频分块大小，默认 240。
            stop_word: 触发终止的关键字。
        """

    def __enter__(self: object) -> object:
        """打开音频流。"""

    def __exit__(self: object, type: object, value: object, traceback: object) -> object:
        """关闭流并释放资源。"""

    def init_queue(self: object, audio_queue: queue.Queue) -> None:
        """
        初始化音频队列。
        参数:
            audio_queue: 保存音频数据的队列。
        """

    def reset(self: object) -> None:
        """
        重置 ASR 客户端状态。
        注意：重置后必须重新初始化队列。
        """

    def generate_pcm(self: object):
        """
        生成器：不断从 track 中产出 pcm 音频数据。
        """

    def listen_print_loop(self: object, responses: object) -> None:
        """
        遍历服务器响应并打印结果。
        - 支持中间结果与最终结果。
        - 中间结果会覆盖前一行。
        - 最终结果会换行保留。
        """

    def send_request(self):
        """
        向 Google ASR 服务发送请求。
        """

    async def generate(self, **kwargs):
        """
        Google ASR 转录。
        参数:
            track: WebRTC 音频轨道。
            output_queue: 转录结果队列，传递给 LLM。
            audio_player: 控制前端播放，用于打断。
            stop_event: 停止信号。
            interrupt_event: 打断信号。
            vad: 语音活动检测 (可选)。
            timeout: 超时时间 (默认 600s)。
            resampler: 音频重采样器 (可选)。
        """



class WhisperASR(ASRClient):
    """
    Whisper ASR 模型。
    """

    def __init__(self, model: str = 'tiny', model_dir: str = None, language_code: str = 'zh', stop_word: str = None, **kwargs):
        """
        初始化 Whisper ASR 模型。
        参数:
            model: 模型名称，默认 tiny。
            model_dir: 模型目录，默认 None。
            language_code: 语言编码，默认 'zh'。
            stop_word: 停止词，默认 None。
        """

    async def generate(self, **kwargs):
        """
        Whisper ASR 转录。
        参数:
            track: WebRTC 音频轨道。
            output_queue: 转录结果队列。
            audio_player: 播放器，用于打断。
            stop_event: 停止信号。
            interrupt_event: 打断信号。
            max_processes: 最大并行进程数，默认 1。
            vad: 语音活动检测 (可选)。
            timeout: 超时，默认 600s。
            resampler: 音频重采样器。
        """



class FunASR(ASRClient):
    """
    使用 Paraformer 模型的流式语音识别。
    """

    def __init__(
        self,
        model_path: str = FUN_ASR_MODEL,
        chunk_size: list = [0, 5, 5],
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        stop_word: str = None,
        **kwargs
    ):
        """
        初始化 ParaformerStreaming 模型。
        参数:
            model_path: 预训练 Paraformer 模型路径。
            chunk_size: [未知, 帧大小, 回看帧数]。
                - chunk_size[0]: 未知，一般为 0。
                - chunk_size[1]: 帧大小（单位：秒）。
                - chunk_size[2]: 编码器/解码器回看帧数。
            encoder_chunk_look_back: 编码器回看块数。
            decoder_chunk_look_back: 解码器回看块数。
        """

    async def generate(self, **kwargs):
        """
        FunASR 转录。
        参数:
            track: WebRTC 音频轨道。
            output_queue: 转录结果队列。
            audio_player: 播放器，用于打断。
            stop_event: 停止信号。
            interrupt_event: 打断信号。
            vad: 语音活动检测 (可选)。
            timeout: 超时，默认 600s。
        """



class BaiduASR(ASRClient):
    """
    百度 ASR 客户端（非流式！）
    """

    def __init__(self, access_token: str, dev_pid: str, uri: str, stop_word: str = None, **kwargs):
        """
        初始化 Baidu ASR 客户端。
        参数:
            access_token: 百度 access token。
            dev_pid: 百度 dev pid。
            uri: 请求地址。
            stop_word: 停止词。
        注意：这是非流式 ASR！
        """

    async def generate(self, **kwargs):
        """
        Baidu ASR 转录。
        参数:
            track: WebRTC 音频轨道。
            output_queue: 转录结果队列。
            audio_player: 播放器，用于打断。
            stop_event: 停止信号。
            interrupt_event: 打断信号。
            vad: 语音活动检测 (可选)。
            timeout: 超时，默认 600s。
            resampler: 音频重采样器。
        """

class ASRClientFactory:
    @staticmethod
    def create(platform, **kwargs) -> ASRClient:
        """
        创建一个 ASR 客户端实例。
        参数:
            platform (str): 要创建的 ASR 客户端所属平台名称。
        返回:
            ASRClient: 创建好的 ASR 客户端实例。
        """
