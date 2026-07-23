# <center> AIRTC </center>

([简体中文](./README_zh.md) | English)

## Summary
AIRTC is a public high-performance, real-time communication framework designed for low-latency audio streaming. It leverages advanced LLM technologies to provide a seamless communication experience AI and human.

[**Summary**](#summary) | 
[**Requirements**](#requirements) | 
[**Quick Start**](#quick-start) | 
[**Client Setup**](#client-setup) | 
[**Access & Password Setup**](#password-setup) 

<a name="requirements"></a>
## Requirements
To run AIRTC, run the requirements.txt
```
pip install -r requirements.txt
```

<a name="quick-start"></a>
## Quick Start
To quickly start using AIRTC, follow these steps:

1. **Clone the repository**:
   ```
   git clone https://github.com/crickwang/AIRTC.git
   cd AIRTC
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the server**:
   ```
   python server.py
   ```

4. **Access the web interface locally**:
   Open your web browser and go to `http://localhost:8081`. This serves the introduction page, from which you can try a guest session or log in.

Or, alternatively, 

```python
from server import WebPage

webpage = WebPage()
webpage.run_web_app()
```
Then, Open your web browser and go to `http://localhost:8081`.

<a name="password-setup"></a>
## Access & Password Setup
Access to AIRTC is controlled by an account/guest system, not a single shared password. On first load, a visitor lands on the introduction page and can either:

- **Try as a guest** — no account needed. Each browser gets a limited number of free trial conversations (3 by default, see `GUEST_CONVERSATION_LIMIT` in `auth_store.py`), tracked via a `guest_token` cookie. Guest transcripts aren't persisted.
- **Sign up / log in** — registered users get a larger quota (30 conversations by default, see `DEFAULT_CONVERSATION_LIMIT` in `auth_store.py`), adjustable per-user via `auth_store.set_conversation_limit(username, new_limit)`. Accounts and conversation history are stored in a local SQLite database (`auth.db` in production, `auth.dev.db` otherwise — controlled by the `APP_ENV` environment variable).

Quota is only checked when a session is actually started (clicking "Start"), not just from loading the page — so a page view alone never counts against anyone's limit.

### Extra: shared site-wide password (optional)
If you want an additional layer on top of the account system — e.g. to keep an early/internal deployment from getting unhandleable traffic before you're ready for public signups — a shared-secret mechanism is still available, though it isn't wired in by default:

1. In your `.env` file (or however you set environment variables), add:
   ```
   SECRET_KEY=<your-own-password>
   ```
2. Add a check against `SECRET_KEY` (see `config/constants.py`) at the top of `offer()` (and/or `introduction()`/`generate()`) in `server.py` — e.g. reject the request unless a submitted password's SHA-256 hash matches `SECRET_KEY`. This isn't present in the current `offer()` implementation; you'd be reintroducing it as an extra gate.
3. Update the HTML templates in `webpage/` to collect and submit the password accordingly.

<a name="client-setup"></a>
## Client Setup
To set up the client for AIRTC, you may follow these steps:
   > note: The client setup is not official, refer to the documentation on the official website if any error happens here. :P

Select which platform each stage uses via CLI flags when starting the server: `--asr` (default `google`), `--llm` (default `google`), `--tts` (default `azure`), `--vad` (default `simple`).

Two kinds of settings are used throughout: **environment variables** (secrets — set via `.env` or your shell) and **`config/config.yaml` keys** (non-secret settings — model names, regions, voices, prompts). Don't confuse the two; setting a config.yaml-only key as an environment variable (or vice versa) will silently be ignored.

- **Google ASR**:
   1. Set up a Google Cloud account and enable the Speech-to-Text API following 
   [this guide for gcloud](https://cloud.google.com/sdk/gcloud) and
   [this guide for STT](https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries).

   2. Create your own cloud project with the STT service enabled.

   3. Create a service account, download its JSON key, and point Application Default Credentials at it:
      ```
      export GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account.json>
      ```
      (The client uses `speech.SpeechClient()` with no API key argument — it authenticates via ADC, not a `GOOGLE_API_KEY`.)

   4. Set the project ID in `config/config.yaml`, not as an environment variable:
      ```yaml
      GOOGLE_PROJ_ID: <your_project_id>
      ```

   5. When initializing the ASR client with Google, pass in necessary parameters such as `language_code`, `rate`, and `chunk_size` in your requests.

- **Baidu ASR**
   1. Go to [Baidu AI](https://ai.baidu.com/tech/speech) and create an account.

   2. Go to [speech](https://console.bce.baidu.com/ai-engine/speech/overview/index) section and create your own application.

   3. With the application, you should have access to the APPID, API key and secret key.
   Add the following environment variables:
   ```
   export BAIDU_APP_ID=<your_app_id>
   export BAIDU_API_KEY=<your_api_key>
   export BAIDU_SECRET_KEY=<your_secret_key>
   ```

   4. In the utils under AIRTC, call baidu() with appropriately set environment variables to get the access token from baidu.
   Add the following environment variable:
   ```
   export BAIDU_ACCESS_TOKEN=<your_access_token>
   ```

   5. Pass in the parameters accordingly. 

- **Baidu LLM** (via Baidu AI Studio)
   1. Go to [Baidu AI Studio](https://aistudio.baidu.com/overview) and create an API key.

   2. Set the API key as an environment variable:
      ```
      export BAIDU_AISTUDIO_API_KEY=<your_api_key>
      ```

   3. The base URL and model come from `config/config.yaml`:
      ```yaml
      BAIDU_AISTUDIO_BASE_URL: https://aistudio.baidu.com/llm/lmapi/v3
      llm_model: <model-name>   # e.g. ernie-4.5-0.3b
      ```

- **Google LLM** (Gemini, via Google AI Studio) — the default `--llm` platform
   1. Create an API key at [Google AI Studio](https://ai.google.dev/).

   2. Set it as an environment variable:
      ```
      export GOOGLE_AI_API_KEY=<your_api_key>
      ```

   3. Optionally choose a model in `config/config.yaml` (see the [model list](https://ai.google.dev/gemini-api/docs/models)):
      ```yaml
      google_llm_model: gemini-2.5-flash-lite
      ```

- **Azure TTS** — the default `--tts` platform
   1. Go to [Azure Portal](https://portal.azure.com/) and create an account.
   2. Create a new Speech service resource.
   3. Get your API key from the Azure portal and set it as an environment variable:
      ```
      export AZURE_TTS_KEY=<your_api_key>
      ```
   4. Set the region and voice in `config/config.yaml` (not environment variables):
      ```yaml
      azure_tts_region: eastus
      azure_tts_voice: zh-CN-YunyangNeural
      ```

- **Edge TTS**
   This is an open source model — no API key needed. Set the voice/language in `config/config.yaml`:
   ```yaml
   edge_tts_voice: zh-CN-YunyangNeural
   edge_tts_language: zh-CN
   ```

- **Google TTS**
   Uses the same Google Cloud credentials as Google ASR (`GOOGLE_APPLICATION_CREDENTIALS`). Set the voice/language in `config/config.yaml`:
   ```yaml
   google_tts_voice: cmn-CN-Chirp3-HD-Puck
   google_tts_language: cmn-CN
   ```

- For more information please contact Email: yiw085@ucsd.edu
