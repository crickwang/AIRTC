# <center> AIRTC </center>

## Summary
AIRTC is a high-performance, real-time communication framework designed for low-latency audio streaming. It leverages advanced LLM technologies to provide a seamless communication experience AI and human.

[**Summary**](#summary) | [**Requirements**](#requirements) | [**Quick Start**](#quick-start)

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

4. **Access the web interface**:
   Open your web browser and go to `http://localhost:8081`.

Or, alternatively, 

```python
from AIRTC.server import WebPage

webpage = WebPage()
webpage.run_web_app()
```
Then, Open your web browser and go to `http://localhost:8081`.