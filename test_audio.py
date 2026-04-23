import base64
import requests
import json
import os
from dotenv import load_dotenv
import glob

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")

def test_audio():
    # Find any mp3 file
    files = glob.glob(r"Deus Ex - train\Deus Ex - train\audio\*.mp3")
    if not files:
        print("No mp3 found")
        return
    file_path = files[0]
    print(f"Testing with: {file_path}")
    
    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Using openrouter with openai/gpt-4o-audio-preview
    # OpenRouter expects standard OpenAI multi-modal syntax for audio
    payload = {
        "model": "google/gemini-2.5-pro",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe this audio message exactly. Output ONLY the raw text transcript."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:audio/mp3;base64,{audio_data}"
                        }
                    }
                ]
            }
        ]
    }
    
    resp = requests.post(url, headers=headers, json=payload)
    print("Status:", resp.status_code)
    try:
        print("Response:", json.dumps(resp.json(), indent=2))
        print("\nTranscript:", resp.json()['choices'][0]['message']['content'])
    except Exception as e:
        print("Error parsing JSON:", resp.text)

if __name__ == "__main__":
    test_audio()
