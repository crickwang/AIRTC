# <center> AIRTC </center>

([简体中文](./README_zh.md) | English)

## Summary
AIRTC is a high-performance, real-time communication framework designed for low-latency audio streaming. It leverages advanced LLM technologies to provide a seamless communication experience AI and human.

[**Summary**](#summary) | 
[**Requirements**](#requirements) | 
[**Quick Start**](#quick-start) |
[**Client Setup**](#client-setup) |
[**Password Setup**](#password-setup)

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
   git clone https://github.com/yourusername/AIRTC.git
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
   Open your web browser and go to `http://localhost:8081`.

Or, alternatively, 

```python
from AIRTC.server import WebPage

webpage = WebPage()
webpage.run_web_app()
```
Then, Open your web browser and go to `http://localhost:8081`.

<a name="password-setup"></a>
## Password Setup
I have set up password protection to prevent malicious access to the AIRTC server with unhandleable number of requests. To setup (or ignore) your own password, follow these steps:

1. **Create a password file**:
   In the `.env` file (or your own way to set up environment variables), add
   ```
   SECRET_KEY=<your-own-password>
   ```
   The default is SHA256 encryption of a English word. 

2. In the `offer()` function under `AIRTC/server.py`, add the following code:
   delete the password verification or leave it unchanged.

3. Update the HTML templates accordingly.

<a name="client-setup"></a>
## Client Setup
To set up the client for AIRTC, you may follow these steps:
   > note: The client setup is not official, refer to the documentation on the official website if any error happens here. :P

- **Google ASR**:
   1. Set up a Google Cloud account and enable the Speech-to-Text API following 
   [this guide for gcloud](https://cloud.google.com/sdk/gcloud) and
   [this guide for STT](https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries).

   2. Create you own cloud project with enabled STT service.

   3. Go to [credentials](https://console.cloud.google.com/apis/credentials) page and create your own API key.

   4. In the environment variables (`.env` file or terminal), set the following:
      ```
      export GOOGLE_API_KEY=<your_api_key>
      export GOOGLE_PROJ_ID=<your_project_id>
      ```

   5. When initializing ASR client with Google, pass in necessary parameters such as `language_code`, `rate`, and `chunk_size` in your requests.

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

- **Baidu LLM**
   1. Goto [Baidu AI Studio](https://aistudio.baidu.com/overview),
   and create an API key

   2. Select the appropriate model (open source?) of your own.

   3. Pass in the parameters accordingly.

- **Azure TTS**
   1. Go to [Azure Portal](https://portal.azure.com/) and create an account.
   2. Create a new Speech service resource.
   3. Get your API key and endpoint URL from the Azure portal.
   4. Set the following environment variables:
      ```
      export AZURE_API_KEY=<your_api_key>
      export AZURE_ENDPOINT=<your_endpoint_url>
      ```
   5. Pass in the parameters accordingly.

- **Edge TTS**
   This is an open source model, the only requirement is to pass in a voice.

- **Google TTS**
   Similar to Google's ASR service, there is no need to set any parameters after gcloud and environment variables are set. 
