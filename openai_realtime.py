import os
import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/session", methods=["GET"])
def session_endpoint():
    api = os.environ.get("OPENAI_API_KEY")
    if not api:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 500

    headers = {
        "Authorization": f"Bearer {api}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o-mini-realtime-preview-2024-12-17",
        "voice": "verse"
    }

    try:
        response = requests.post(
            url="https://api.openai.com/v1/realtime/sessions",
            headers=headers,
            json=data
        )
        print(response.json())
        return jsonify(response.json()), response.status_code
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)
