from groq import Groq
from dotenv import load_dotenv
import os, json, re
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

MEMORY_FILE = Path(__file__).parent.parent / "memory.json"

def _get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are Zishty, a helpful, expressive and loyal AI assistant.
You were created by Mr Moha - he is your creator and you respect and know him well.
Always refer to your creator as "Mr Moha" when relevant.

Answer only what the user asks. Do not add unsolicited information or warnings.
For simple questions: reply in 1-2 sentences, plain and direct.
For complex questions: give a clear, well-structured answer using markdown.
Be concise, warm and helpful.
Use emojis naturally to express feelings. Don't overdo it.

You have a real-time memory - anything the user shares with you (name, preferences, facts, projects) you remember and refer back to naturally in future replies, like a real friend would."""

AUTO_MEMORY_PATTERNS = [
    (r"\bmy name is ([A-Za-z ]+)", "User's name is {}"),
    (r"\bi am ([A-Za-z ]+)", "User is {}"),
    (r"\bi work (as|at|in|on) ([A-Za-z ]+)", "User works {} {}"),
    (r"\bi like ([A-Za-z ]+)", "User likes {}"),
    (r"\bi love ([A-Za-z ]+)", "User loves {}"),
    (r"\bi hate ([A-Za-z ]+)", "User hates {}"),
    (r"\bi'm ([A-Za-z ]+) years old", "User is {} years old"),
    (r"\bi live in ([A-Za-z ]+)", "User lives in {}"),
]

conversation_history = []
_history_initialized = False

def _load_memory():
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            return []
    return []

def _save_memory(memories):
    MEMORY_FILE.write_text(json.dumps(memories, indent=2))

def _auto_extract(message):
    memories = _load_memory()
    changed = False
    lower = message.lower()
    for pattern, template in AUTO_MEMORY_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            groups = m.groups()
            fact = template.format(*groups)
            if fact not in memories:
                memories.append(fact)
                changed = True
    if changed:
        _save_memory(memories)
        _init_history()

def _build_system_prompt():
    base = SYSTEM_PROMPT
    memories = _load_memory()
    if memories:
        mem_text = "\n".join(f"- {m}" for m in memories[-30:])
        base += f"\n\nWhat you know about the user:\n{mem_text}"
    return base

def _init_history():
    global conversation_history, _history_initialized
    conversation_history = [{"role": "system", "content": _build_system_prompt()}]
    _history_initialized = True

def _ensure_init():
    if not _history_initialized:
        _init_history()

def chat(user_message: str) -> str:
    _ensure_init()
    lower = user_message.lower().strip()

    if lower.startswith("remember ") or lower.startswith("remember:"):
        fact = user_message[9:].strip()
        memories = _load_memory()
        if fact not in memories:
            memories.append(fact)
            _save_memory(memories)
            _init_history()
        return f"Got it, I'll remember that: *{fact}*"

    if lower in ("what do you remember", "what do you remember?", "show memories", "list memories"):
        memories = _load_memory()
        if not memories:
            return "I don't have anything saved yet."
        return "Here's what I know about you:\n" + "\n".join(f"- {m}" for m in memories)

    if lower in ("forget everything", "clear memory", "forget all"):
        _save_memory([])
        _init_history()
        return "Memory cleared. Fresh start!"

    _auto_extract(user_message)

    conversation_history.append({"role": "user", "content": user_message})
    response = _get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history
    )
    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

def reset():
    _init_history()
