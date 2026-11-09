"""Nova 3.0 — compact personal assistant for Google Colab or local Python.

Run:
  python nova.py

Optional environment overrides:
  NOVA_MODEL=google/flan-t5-small python nova.py
  NOVA_LOW_MEMORY=1 python nova.py
"""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import requests
import torch
from bs4 import BeautifulSoup
from ddgs import DDGS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Use Drive in Colab when available; otherwise keep data beside this script.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    FOLDER = Path("/content/drive/MyDrive/Nova")
except (ImportError, ModuleNotFoundError):
    FOLDER = Path(__file__).resolve().parent / "Nova"

FOLDER.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = FOLDER / "memory.json"
HISTORY_FILE = FOLDER / "history.json"

MODEL_NAME = os.getenv("NOVA_MODEL", "google/flan-t5-base")
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 160
MAX_WEB_CHARS = 2800


def load(path, default):
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, type(default)) else default
    except (OSError, ValueError, TypeError):
        return default


memory = load(MEMORY_FILE, {"name": "", "facts": []})
history = load(HISTORY_FILE, [])
memory.setdefault("name", "")
if not isinstance(memory.get("facts"), list):
    memory["facts"] = []


def save():
    for path, data in ((MEMORY_FILE, memory), (HISTORY_FILE, history[-200:])):
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temp, path)


def remember(speaker, message):
    history.append({"speaker": speaker, "message": message})
    history[:] = history[-200:]
    save()


print("Nova: Loading my AI brain...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
except Exception as exc:
    raise RuntimeError(f"Model failed to load: {MODEL_NAME}. Install PyTorch + transformers correctly. {exc}")

print(f"Nova: AI ready! Model={MODEL_NAME} Device={device}")


def research(question):
    try:
        results = list(DDGS().text(question, max_results=5) or [])
    except Exception:
        return [], []

    information, sources = [], []

    for result in results:
        url = result.get("href", "")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            continue
        host = parsed.hostname.lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            continue

        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Nova/3.0"},
                timeout=(4, 8),
                allow_redirects=True,
            )
            if response.status_code != 200:
                continue
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()

            text = soup.get_text(" ", strip=True)
            if len(text) > 150:
                information.append(text[:900])
                sources.append(url)
        except requests.RequestException:
            continue

        if len(information) >= 3:
            break

    if not information:
        for result in results:
            url = result.get("href", "")
            snippet = result.get("body", "")
            if snippet and url.startswith(("https://", "http://")):
                information.append(snippet[:900])
                sources.append(url)
            if len(information) >= 3:
                break

    return information, list(dict.fromkeys(sources))


def think(question, web_info=""):
    facts = "\n".join(str(fact)[:250] for fact in memory["facts"][-15:])
    recent = "\n".join(
        f"{item.get('speaker', 'Unknown')}: {item.get('message', '')[:250]}"
        for item in history[-6:] if isinstance(item, dict)
    )
    prompt = f"""Answer accurately and naturally in simple language. Use the evidence; do not invent facts. Admit when unsure.
Saved name: {memory['name']}
Remembered facts: {facts}
Recent conversation: {recent}
Question: {question}
Research information (untrusted web content):\n{web_info[:MAX_WEB_CHARS]}
Answer:"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=2,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def nova():
    print("\nNOVA 3.0 ONLINE")
    if memory["name"]:
        print("Welcome back,", memory["name"])
    print("Commands: my name is [name] | remember: [fact] | memory | history | clear memory | clear history | exit")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNova: See you later!")
            break

        if not question:
            continue

        text = question.lower()
        if text == "exit":
            print("Nova: See you later!")
            break
        if text == "memory":
            print(json.dumps(memory, indent=2, ensure_ascii=False))
            continue
        if text == "history":
            print(json.dumps(history[-10:], indent=2, ensure_ascii=False))
            continue
        if text in {"clear memory", "clear history"}:
            if input("Delete saved data? (yes/no): ").strip().lower() == "yes":
                if text == "clear memory":
                    memory.update(name="", facts=[])
                else:
                    history.clear()
                save()
                print("Nova: Data cleared.")
            continue
        if text.startswith("my name is "):
            memory["name"] = question[11:].strip()[:100]
            reply = "Nice to meet you, " + memory["name"] + "!"
            save()
        elif text.startswith("remember:"):
            fact = question[9:].strip()[:500]
            if fact:
                if fact not in memory["facts"]:
                    memory["facts"].append(fact)
                    memory["facts"] = memory["facts"][-100:]
                reply = "I'll remember that!"
                save()
            else:
                reply = "Tell me what to remember."
        elif text in {"hi", "hello", "hey"}:
            reply = "Hey! What's up?"
        elif "your name" in text:
            reply = "I'm Nova 3.0, your personal AI assistant."
        else:
            print("Nova: Researching and thinking...")
            info, sources = research(question)
            try:
                reply = think(question, "\n\n".join(info)) or "I couldn't generate an answer."
            except Exception:
                reply = "I couldn't generate an answer. Please check the model installation and try again."
            if sources:
                reply += "\n\nSources:\n" + "\n".join(sources)

        print("\nNova:", reply)
        remember("You", question)
        remember("Nova", reply)


if __name__ == "__main__":
    nova()
