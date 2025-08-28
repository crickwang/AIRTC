# <center> AIRTC </center>

(简体中文 | [English](./README.md))

## 摘要
AIRTC 是一个高性能、实时通信框架，专为低延迟音频流设计。它利用先进的 LLM 技术，为 AI 与人类之间提供无缝的沟通体验。
   > 中文版仅供参考，请以英文版为准。

[**摘要**](#summary) | 
[**环境要求**](#requirements) | 
[**快速开始**](#quick-start) |
[**客户端设置**](#client-setup) |
[**密码设置**](#password-setup)

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
   git clone https://github.com/yourusername/AIRTC.git
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
   打开浏览器，访问 `http://localhost:8081`。

或者，使用 Python 调用： 

```python
from AIRTC.server import WebPage

webpage = WebPage()
webpage.run_web_app()
```

然后在浏览器中访问 `http://localhost:8081`。

<a name="password-setup"></a>
## 密码设置
我在 AIRTC 服务器端加了密码保护，以防止恶意访问导致无法处理的请求数量。要设置（或忽略）你自己的密码，请按以下步骤操作：

1. **创建密码文件**:
   在 `.env` 文件中（或用你自己的方式设置环境变量），添加：
   ```bash
   SECRET_KEY=<your-own-password>
   ```
   默认是某个英文单词的 SHA256 加密结果。 

2. 在 `AIRTC/server.py` 中的 `offer()` 函数里，删除或保留密码验证逻辑。

3. 更新 HTML 模板。

<a name="client-setup"></a>
## 客户端设置
要为 AIRTC 设置客户端，可以按照以下步骤操作：  
> 注意：客户端设置不是官方流程，如果出现错误请参考官方文档。:P

- **Google ASR**:
   1. 注册 Google Cloud 账号，并启用 Speech-to-Text API，参考：
      [gcloud 指南](https://cloud.google.com/sdk/gcloud) 和
      [STT 指南](https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries)。
   2. 创建启用了 STT 服务的云项目。
   3. 前往 [credentials](https://console.cloud.google.com/apis/credentials) 页面创建 API key。
   4. 在环境变量中设置：
      ```bash
      export GOOGLE_API_KEY=<your_api_key>
      export GOOGLE_PROJ_ID=<your_project_id>
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

- **Baidu LLM**
   1. 前往 [Baidu AI Studio](https://aistudio.baidu.com/overview) 创建 API key。
   2. 选择合适的模型（开源？）。
   3. 按需传入参数。

- **Azure TTS**
   1. 前往 [Azure Portal](https://portal.azure.com/) 创建账号。
   2. 创建新的 Speech service 资源。
   3. 从 Azure portal 获取 API key 和 endpoint URL。
   4. 设置环境变量：
      ```bash
      export AZURE_API_KEY=<your_api_key>
      export AZURE_ENDPOINT=<your_endpoint_url>
      ```
   5. 按需传入参数。

- **Edge TTS**
   开源模型，仅需传入语音参数。

- **Google TTS**
   与 Google ASR 类似，配置好 gcloud 和环境变量后无需额外参数。
