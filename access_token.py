
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
client_id = os.getenv('BAIDU_API_KEY')
client_secret = os.getenv('BAIDU_SECRET_KEY')

def main():
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}"
    payload = ""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()['access_token']
    
if __name__ == '__main__':
    main()