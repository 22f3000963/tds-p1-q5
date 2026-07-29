import os
import json
import time
import sys
import io
import traceback
import threading
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
from datetime import datetime, timezone

app = FastAPI()

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url="https://aipipe.org/openrouter/v1"
)
LOG_FILE = "run.jsonl"
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# In-memory history for multi-turn conversations
chat_histories = {}

def log_event(event_dict):
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event_dict) + "\n")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/run.jsonl")
def get_log():
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, "a").close()
    return FileResponse(LOG_FILE, media_type="application/jsonlines+json")

def execute_python(code: str) -> str:
    """Executes python code and returns stdout."""
    # Add imports for convenience
    full_code = f"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json

{code}
"""
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        exec(full_code, {})
        sys.stdout = old_stdout
        out = redirected_output.getvalue()
        if not out:
            out = "Code executed successfully, no output."
        return out[:8000]
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error executing code:\n{traceback.format_exc()}"[:8000]

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Executes Python code server-side and returns stdout. Use this to download datasets, process data with pandas, or scrape websites. You must use print() to see the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Can use pandas, numpy, requests, bs4."
                    }
                },
                "required": ["code"]
            }
        }
    }
]

def process_message(chat_id, text):
    start_time = time.time()
    
    if chat_id not in chat_histories:
        chat_histories[chat_id] = [
            {"role": "system", "content": "You are a data-analyst Telegram bot. You MUST reply with exactly ONE JSON object matching the shape requested by the user. If they specify `{\"answer\": ...}`, you must strictly output a JSON object containing that. Do NOT wrap the JSON in markdown code blocks like ```json. Do NOT include any extra text. Use the `run_python` tool to download and analyze datasets. Never guess a number you can compute. If it's a multi-turn conversation, answer the latest message."}
        ]
        
    chat_histories[chat_id].append({"role": "user", "content": text})
    # Keep last 20 messages to avoid context overflow
    if len(chat_histories[chat_id]) > 21:
        chat_histories[chat_id] = [chat_histories[chat_id][0]] + chat_histories[chat_id][-20:]
        
    messages = chat_histories[chat_id].copy()
    
    log_event({"actor": "system", "event": "processing_message", "chat_id": chat_id, "text": text})

    for _ in range(10): # max 10 steps
        if time.time() - start_time > 210: # budget ~ 210s
            break
            
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o",
                messages=messages,
                tools=tools,
                temperature=0.0
            )
        except Exception as e:
            return json.dumps({"answer": f"OpenAI error: {str(e)}", "log_url": f"{BASE_URL}/run.jsonl"})
            
        msg = response.choices[0].message
        messages.append(msg)
        
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "run_python":
                    args = json.loads(tool_call.function.arguments)
                    code = args.get("code", "")
                    
                    log_event({"actor": "agent", "event": "tool_call", "code": code})
                    
                    output = execute_python(code)
                    
                    log_event({"actor": "system", "event": "tool_result", "output": output})
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "run_python",
                        "content": output
                    })
        else:
            # We got a text response!
            reply = msg.content.strip()
            
            # Clean markdown if present
            if reply.startswith("```json"):
                reply = reply[7:]
            if reply.startswith("```"):
                reply = reply[3:]
            if reply.endswith("```"):
                reply = reply[:-3]
            reply = reply.strip()
            
            try:
                parsed = json.loads(reply)
                if "answer" not in parsed:
                    parsed = {"answer": parsed}
                parsed["log_url"] = f"{BASE_URL}/run.jsonl"
                
                final_reply = json.dumps(parsed)
                chat_histories[chat_id].append({"role": "assistant", "content": final_reply})
                log_event({"actor": "agent", "event": "final_reply", "reply": final_reply})
                return final_reply
                
            except Exception as e:
                # If they didn't output pure JSON, wrap whatever they said
                safe_reply = json.dumps({"answer": reply, "log_url": f"{BASE_URL}/run.jsonl"})
                chat_histories[chat_id].append({"role": "assistant", "content": safe_reply})
                log_event({"actor": "agent", "event": "final_reply", "reply": safe_reply, "note": "Failed to parse JSON, wrapped string."})
                return safe_reply
                
    # Timeout / Max steps reached
    fallback = json.dumps({"answer": "timeout or max steps reached", "log_url": f"{BASE_URL}/run.jsonl"})
    chat_histories[chat_id].append({"role": "assistant", "content": fallback})
    return fallback


def send_message(chat_id, text):
    url = f"{TELEGRAM_API}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)

def telegram_poller():
    offset = 0
    while True:
        try:
            url = f"{TELEGRAM_API}/getUpdates?timeout=30&offset={offset}"
            resp = requests.get(url, timeout=40)
            data = resp.json()
            if data.get("ok"):
                for update in data["result"]:
                    offset = update["update_id"] + 1
                    
                    if "message" in update and "text" in update["message"]:
                        chat_id = update["message"]["chat"]["id"]
                        text = update["message"]["text"]
                        
                        # Process in background so we don't block poller
                        def worker(c_id, txt):
                            reply = process_message(c_id, txt)
                            send_message(c_id, reply)
                            
                        threading.Thread(target=worker, args=(chat_id, text)).start()
        except Exception as e:
            time.sleep(2)

def pinger():
    """Keeps the Render instance alive"""
    while True:
        try:
            requests.get(f"{BASE_URL}/health", timeout=5)
        except:
            pass
        time.sleep(600) # 10 minutes

@app.on_event("startup")
def startup_event():
    if not BOT_TOKEN:
        print("WARNING: BOT_TOKEN not set!")
    else:
        threading.Thread(target=telegram_poller, daemon=True).start()
    threading.Thread(target=pinger, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    # run locally
    uvicorn.run(app, host="0.0.0.0", port=8000)
