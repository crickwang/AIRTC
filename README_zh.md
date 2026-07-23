# <center> AIRTC </center>

(简体中文 | [English](./README.md))

## 摘要
AIRTC 是一个高性能、实时通信框架，专为低延迟音频流设计。它利用先进的 LLM 技术，为 AI 与人类之间提供无缝的沟通体验。
   > 中文版仅供参考，请以英文版为准。

[**摘要**](#summary) | 
[**环境要求**](#requirements) | 
[**快速开始**](#quick-start) |
[**客户端设置**](#client-setup) |
[**访问与密码设置**](#password-setup)

<a name="requirements"></a>
## 环境要求
运行 AIRTC，需要执行以下命令安装依赖：
```
pip install -r requirements.txt
```

<a name="quick-start"></a>
## 快速开始
要快速启动 AIRTC，请按照以下步骤操作：

1. **克隆仓库**:
   ```bash
   git clone https://github.com/crickwang/AIRTC.git
   cd AIRTC
   ```

2. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

3. **运行服务器**:
   ```bash
   python server.py
   ```

4. **本地访问网页界面**:
   打开浏览器，访问 `http://localhost:8081`。这里会展示介绍页面，你可以选择以访客身份试用，或登录账号。

或者，使用 Python 调用： 

```python
from server import WebPage

webpage = WebPage()
webpage.run_web_app()
```

然后在浏览器中访问 `http://localhost:8081`。

<a name="password-setup"></a>
## 访问与密码设置
AIRTC 的访问控制基于账号/访客系统，而不是单一的共享密码。访问者首先会看到介绍页面，可以选择：

- **以访客身份试用** — 无需账号。每个浏览器默认有 3 次免费试用对话（见 `auth_store.py` 中的 `GUEST_CONVERSATION_LIMIT`），通过 `guest_token` cookie 追踪，访客的对话记录不会被持久化保存。
- **注册 / 登录** — 注册用户默认拥有更大的配额（默认 30 次对话，见 `auth_store.py` 中的 `DEFAULT_CONVERSATION_LIMIT`），可通过 `auth_store.set_conversation_limit(username, new_limit)` 按用户调整。账号和对话记录保存在本地 SQLite 数据库中（生产环境为 `auth.db`，其他情况下为 `auth.dev.db`，由环境变量 `APP_ENV` 控制）。

配额只在真正开始一次会话时（点击"开始"）才会被扣除，仅仅打开页面并不会消耗任何人的配额。

### 额外：全站共享密码（可选）
如果你想在账号系统之上再加一层保护——例如在正式开放注册前，防止早期/内部部署收到无法处理的大量请求——仍然可以使用共享密钥机制，但它默认并未接入：

1. 在 `.env` 文件中（或用你自己的方式设置环境变量），添加：
   ```
   SECRET_KEY=<your-own-password>
   ```
2. 在 `server.py` 的 `offer()`（以及/或 `introduction()`、`generate()`）函数开头，加入对 `SECRET_KEY`（见 `config/constants.py`）的校验逻辑——例如：仅当提交密码的 SHA-256 哈希值与 `SECRET_KEY` 匹配时才放行请求。当前 `offer()` 的实现中并没有这段逻辑，你需要自己把它作为额外的一道关卡加回去。
3. 相应地更新 `webpage/` 下的 HTML 模板，用于收集并提交密码。

<a name="client-setup"></a>
## 客户端设置
要为 AIRTC 设置客户端，可以按照以下步骤操作：  
> 注意：客户端设置不是官方流程，如果出现错误请参考官方文档。:P

启动服务器时可通过命令行参数选择每个环节使用的平台：`--asr`（默认 `google`）、`--llm`（默认 `google`）、`--tts`（默认 `azure`）、`--vad`（默认 `simple`）。

全文用到两类配置：**环境变量**（密钥类，通过 `.env` 或 shell 设置）和 **`config/config.yaml` 中的键**（非密钥设置，如模型名称、区域、语音、提示词等）。两者不要混淆——把一个只属于 config.yaml 的键当作环境变量设置（反之亦然）会被静默忽略。

- **Google ASR**:
   1. 注册 Google Cloud 账号，并启用 Speech-to-Text API，参考：
      [gcloud 指南](https://cloud.google.com/sdk/gcloud) 和
      [STT 指南](https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries)。
   2. 创建启用了 STT 服务的云项目。
   3. 创建一个服务账号，下载其 JSON 密钥文件，并让 Application Default Credentials 指向它：
      ```bash
      export GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account.json>
      ```
      （客户端调用 `speech.SpeechClient()` 时不传任何 API key 参数——认证方式是 ADC，而不是 `GOOGLE_API_KEY`。）
   4. 在 `config/config.yaml` 中设置项目 ID，而不是作为环境变量：
      ```yaml
      GOOGLE_PROJ_ID: <your_project_id>
      ```
   5. 初始化 Google ASR 客户端时，传入必要参数，如 `language_code`、`rate` 和 `chunk_size`。

- **Baidu ASR**
   1. 前往 [百度 AI](https://ai.baidu.com/tech/speech) 注册账号。
   2. 前往 [语音服务](https://console.bce.baidu.com/ai-engine/speech/overview/index) 创建应用。
   3. 获取 APPID、API key 和 secret key，并设置环境变量：
      ```bash
      export BAIDU_APP_ID=<your_app_id>
      export BAIDU_API_KEY=<your_api_key>
      export BAIDU_SECRET_KEY=<your_secret_key>
      ```
   4. 在 AIRTC 的 utils 中调用 `baidu()`，用以上变量获取 access token，并设置：
      ```bash
      export BAIDU_ACCESS_TOKEN=<your_access_token>
      ```
   5. 按需传入参数。

- **Baidu LLM**（通过 Baidu AI Studio）
   1. 前往 [Baidu AI Studio](https://aistudio.baidu.com/overview) 创建 API key。
   2. 将 API key 设置为环境变量：
      ```bash
      export BAIDU_AISTUDIO_API_KEY=<your_api_key>
      ```
   3. base URL 和模型名称来自 `config/config.yaml`：
      ```yaml
      BAIDU_AISTUDIO_BASE_URL: https://aistudio.baidu.com/llm/lmapi/v3
      llm_model: <model-name>   # 例如 ernie-4.5-0.3b
      ```

- **Google LLM**（Gemini，通过 Google AI Studio）—— `--llm` 的默认平台
   1. 前往 [Google AI Studio](https://ai.google.dev/) 创建 API key。
   2. 将其设置为环境变量：
      ```bash
      export GOOGLE_AI_API_KEY=<your_api_key>
      ```
   3. 可选：在 `config/config.yaml` 中选择模型（参见[模型列表](https://ai.google.dev/gemini-api/docs/models)）：
      ```yaml
      google_llm_model: gemini-2.5-flash-lite
      ```

- **Azure TTS** —— `--tts` 的默认平台
   1. 前往 [Azure Portal](https://portal.azure.com/) 创建账号。
   2. 创建新的 Speech service 资源。
   3. 从 Azure portal 获取 API key，并设置为环境变量：
      ```bash
      export AZURE_TTS_KEY=<your_api_key>
      ```
   4. 在 `config/config.yaml` 中设置区域和语音（而不是环境变量）：
      ```yaml
      azure_tts_region: eastus
      azure_tts_voice: zh-CN-YunyangNeural
      ```

- **Edge TTS**
   开源模型，无需 API key。在 `config/config.yaml` 中设置语音/语言：
   ```yaml
   edge_tts_voice: zh-CN-YunyangNeural
   edge_tts_language: zh-CN
   ```

- **Google TTS**
   使用与 Google ASR 相同的 Google Cloud 凭证（`GOOGLE_APPLICATION_CREDENTIALS`）。在 `config/config.yaml` 中设置语音/语言：
   ```yaml
   google_tts_voice: cmn-CN-Chirp3-HD-Puck
   google_tts_language: cmn-CN
   ```

- 如需更多信息，请联系邮箱：yiw085@ucsd.edu
