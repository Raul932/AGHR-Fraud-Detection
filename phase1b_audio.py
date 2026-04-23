import base64
import json
import os
import glob
from pathlib import Path
import requests
from datetime import datetime
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

def transcribe_audio_openrouter(file_path: Path) -> str:
    safe_name = file_path.name.encode('ascii', 'replace').decode('ascii')
    print(f"  [Audio] Transcribing: {safe_name}...")
    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")
        
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://reply-eye.example.com",
        "X-Title": "Agentic Fraud Detection"
    }
    
    payload = {
        "model": "google/gemini-2.5-pro",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Transcribe this audio exactly. Just provide the raw text, no commentary."
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
    if resp.status_code == 200:
        return resp.json()['choices'][0]['message']['content'].strip()
    else:
        err = resp.text
        print(f"  [Audio] API Error ({resp.status_code}): {err}")
        try:
            err_json = resp.json()
            if "guardrail" in err_json.get("error", {}).get("message", "").lower():
                print("  [Audio] FATAL: Your OpenRouter API key blocks audio models due to privacy settings.")
                print("  [Audio] Please go to https://openrouter.ai/settings/privacy and allow 'Data logging'.")
        except: pass
        return "[TRANSCRIPTION FAILED OR BLOCKED]"

def process_audio_directory(dataset_dir: str):
    base_dir = Path(dataset_dir)
    audio_dir = base_dir / "audio"
    if not audio_dir.exists() or not audio_dir.is_dir():
        print(f"[Phase 1b] No 'audio' directory found in {dataset_dir}. Skipping audio processing.")
        return

    calls_json_path = base_dir / "calls.json"
    
    # Load existing to resume
    existing_calls = {}
    if calls_json_path.exists():
        with open(calls_json_path, 'r', encoding='utf-8') as f:
            try:
                for c in json.load(f):
                    existing_calls[c['filename']] = c
            except: pass

    mp3_files = list(audio_dir.glob("*.mp3"))
    print(f"\n============================================================")
    print(f"Phase 1b: Audio Transcription ({len(mp3_files)} files)")
    print(f"============================================================")
    
    calls = []
    
    for mf in mp3_files:
        fn = mf.name
        
        # Format: 20870117_010505-guido_doÌˆhn.mp3
        # Extract Timestamp and Name portion
        try:
            ts_str, name_part = fn.split("-", 1)
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").strftime("%Y-%m-%dT%H:%M:%S")
            # name_part might be guido_doÌˆhn.mp3 -> remove .mp3 and replace _ with space
            raw_name = name_part.replace(".mp3", "")
        except ValueError:
            print(f"  [Audio] Warning: Could not parse filename format for {fn}")
            ts = "2087-01-01T00:00:00"
            raw_name = fn.replace(".mp3", "")

        if fn in existing_calls and "[TRANSCRIPTION FAILED" not in existing_calls[fn].get("text", ""):
            calls.append(existing_calls[fn])
            continue
            
        transcript = transcribe_audio_openrouter(mf)
        
        call_record = {
            "filename": fn,
            "ts": ts,
            "raw_name_identifier": raw_name,
            "text": transcript
        }
        calls.append(call_record)
        
        # Save incrementally
        with open(calls_json_path, 'w', encoding='utf-8') as f:
            json.dump(calls, f, indent=2, ensure_ascii=False)

    failed = sum(1 for c in calls if "[TRANSCRIPTION FAILED" in c["text"])
    print(f"[Phase 1b] Audio processing complete. {len(calls)} calls recorded. ({failed} failed)")

if __name__ == "__main__":
    import sys
    d_dir = sys.argv[1] if len(sys.argv) > 1 else "Deus Ex - train/Deus Ex - train"
    process_audio_directory(d_dir)
