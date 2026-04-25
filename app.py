from flask import Flask, request, jsonify, send_from_directory
from assistant.brain import chat, reset
from groq import Groq
from dotenv import load_dotenv
import os, tempfile, requests, base64, hashlib, subprocess, re, difflib, socket, json, sys

# support PyInstaller bundle
# support PyInstaller bundle
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))

def get_groq():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.after_request
def add_headers(response):
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return send_from_directory("static", "chat.html")

@app.route("/sw.js")
def sw():
    response = send_from_directory("static", "sw.js", mimetype="application/javascript")
    response.headers["Service-Worker-Allowed"] = "/"
    return response



    return send_from_directory("static", "index.html")



# ── Chat ──
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400
        return jsonify({"reply": chat(message)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset_endpoint():
    reset()
    return jsonify({"status": "ok"})


# ── Voice: STT (Whisper via Groq) ──
@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio"}), 400
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        with open(tmp.name, "rb") as f:
            result = get_groq().audio.transcriptions.create(model="whisper-large-v3", file=f)
    os.unlink(tmp.name)
    return jsonify({"text": result.text})


# ── Voice: TTS (browser-side via Web Speech API — no server needed) ──
@app.route("/tts", methods=["POST"])
def tts_endpoint():
    # TTS is handled client-side via Web Speech API
    return jsonify({"status": "client-side"})


# ── Intent classifier ──
@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"intent": "chat"})
        r = get_groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": (
                "Classify this user message into exactly one of: image, code, readfile, savefile, createfile, websearch, chat.\n"
                "Reply with ONLY the single word.\n\n"
                f"Message: {message}"
            )}],
            max_tokens=5
        )
        intent = r.choices[0].message.content.strip().lower()
        if intent not in ("image", "code", "readfile", "savefile", "createfile", "websearch", "chat"):
            intent = "chat"
        return jsonify({"intent": intent})
    except Exception as e:
        return jsonify({"intent": "chat", "error": str(e)})


# ── Filesystem ──
@app.route("/fs/list", methods=["POST"])
def fs_list():
    data = request.get_json()
    path = data.get("path", os.path.expanduser("~"))
    try:
        path = os.path.abspath(path)
        items = []
        for entry in sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower())):
            try:
                items.append({
                    "name": entry.name,
                    "path": entry.path,
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else 0
                })
            except PermissionError:
                pass
        return jsonify({"path": path, "items": items, "parent": str(os.path.dirname(path))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fs/read-image", methods=["POST"])
def fs_read_image():
    data = request.get_json()
    path = data.get("path", "")
    try:
        import mimetypes
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        return jsonify({"url": f"data:{mime};base64,{img_b64}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



    data = request.get_json()
    path = data.get("path", "")
    try:
        size = os.path.getsize(path)
        if size > 500_000:
            return jsonify({"error": "File too large to preview"}), 400
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return jsonify({"content": content, "path": path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fs/write", methods=["POST"])
def fs_write():
    data = request.get_json()
    path = data.get("path", "").strip()
    content = data.get("content", "")
    if not path:
        return jsonify({"error": "No path"}), 400
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return jsonify({"ok": True, "path": os.path.abspath(path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fs/create", methods=["POST"])
def fs_create():
    data = request.get_json()
    name = data.get("name", "").strip()
    folder = data.get("folder", "").strip()
    content = data.get("content", "")
    if not name:
        return jsonify({"error": "No filename"}), 400
    path = os.path.join(folder, name) if folder else name
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return jsonify({"ok": True, "path": os.path.abspath(path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Search in files ──
@app.route("/fs/search", methods=["POST"])
def fs_search():
    data = request.get_json()
    query = data.get("query", "").strip()
    path = data.get("path", os.path.expanduser("~"))
    if not query:
        return jsonify({"error": "No query"}), 400
    results = []
    try:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if query.lower() in line.lower():
                                results.append({"file": fpath, "line": i, "text": line.rstrip()})
                                if len(results) >= 100:
                                    return jsonify({"results": results, "truncated": True})
                except Exception:
                    pass
        return jsonify({"results": results, "truncated": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Diff viewer ──
@app.route("/diff", methods=["POST"])
def diff_endpoint():
    data = request.get_json()
    a = data.get("a", "")
    b = data.get("b", "")
    label_a = data.get("label_a", "File A")
    label_b = data.get("label_b", "File B")
    diff = list(difflib.unified_diff(
        a.splitlines(keepends=True),
        b.splitlines(keepends=True),
        fromfile=label_a, tofile=label_b
    ))
    return jsonify({"diff": "".join(diff)})


# ── Code runner ──
@app.route("/run-code", methods=["POST"])
def run_code():
    data = request.get_json()
    code = data.get("code", "").strip()
    lang = data.get("lang", "python").lower()
    if not code:
        return jsonify({"error": "No code"}), 400
    try:
        if lang == "python":
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(code)
                fname = f.name
            result = subprocess.run(["python", fname], capture_output=True, text=True, timeout=10)
            os.unlink(fname)
            return jsonify({"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})
        elif lang in ("javascript", "js", "node"):
            with tempfile.NamedTemporaryFile(suffix=".js", delete=False, mode="w") as f:
                f.write(code)
                fname = f.name
            result = subprocess.run(["node", fname], capture_output=True, text=True, timeout=10)
            os.unlink(fname)
            return jsonify({"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})
        else:
            return jsonify({"error": f"Language '{lang}' not supported. Use python or javascript."}), 400
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Execution timed out (10s limit)"}), 408
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Image: AI edit (remove bg, fix via prompt) ──
@app.route("/image-edit-ai", methods=["POST"])
def image_edit_ai():
    data = request.get_json()
    image_b64 = data.get("image", "")
    instruction = data.get("instruction", "").strip()
    if not image_b64 or not instruction:
        return jsonify({"error": "No image or instruction"}), 400
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        response = get_groq().chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"The user wants to edit this image with the instruction: '{instruction}'. Describe in detail what the edited image should look like, as a vivid image generation prompt. Return only the prompt."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            max_tokens=200
        )
        new_prompt = response.choices[0].message.content.strip()
        # generate the edited image
        gen = requests.post(
            "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
            headers=HF_HEADERS,
            json={"inputs": new_prompt},
            timeout=60
        )
        if gen.status_code != 200:
            return jsonify({"error": f"Generation failed: {gen.text[:200]}"}), 500
        img_b64 = base64.b64encode(gen.content).decode("utf-8")
        return jsonify({"url": f"data:image/jpeg;base64,{img_b64}", "prompt": new_prompt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Image: Remove background ──
@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    data = request.get_json()
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"error": "No image"}), 400
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(image_b64)
        # Use rembg if available, else return error
        try:
            from rembg import remove
            result = remove(img_bytes)
            out_b64 = base64.b64encode(result).decode("utf-8")
            return jsonify({"url": f"data:image/png;base64,{out_b64}"})
        except ImportError:
            return jsonify({"error": "rembg not installed. Run: pip install rembg"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt"}), 400
    enhanced = _enhance_prompt(prompt, "image")
    response = requests.post(
        "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
        headers=HF_HEADERS,
        json={"inputs": enhanced},
        timeout=60
    )
    if response.status_code != 200:
        return jsonify({"error": f"Image generation failed: {response.text[:200]}"}), 500
    img_b64 = base64.b64encode(response.content).decode("utf-8")
    return jsonify({"url": f"data:image/jpeg;base64,{img_b64}", "enhanced_prompt": enhanced})


# ── Image to text ──
@app.route("/image-to-text", methods=["POST"])
def image_to_text():
    data = request.get_json()
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"error": "No image"}), 400
    # Strip data URL prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        response = get_groq().chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            max_tokens=500
        )
        return jsonify({"description": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Sketch to image ──
@app.route("/sketch-to-image", methods=["POST"])
def sketch_to_image():
    data = request.get_json()
    sketch_b64 = data.get("sketch", "")
    prompt = data.get("prompt", "").strip()
    if not sketch_b64:
        return jsonify({"error": "No sketch"}), 400
    if "," in sketch_b64:
        sketch_b64 = sketch_b64.split(",", 1)[1]
    try:
        # Step 1: use vision model to describe what's in the sketch
        vision = get_groq().chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This is a rough sketch. Describe exactly what is drawn in it — shapes, objects, layout, any text or letters visible. Be precise and literal. User also says: '{prompt}'. Create a detailed image generation prompt that faithfully reproduces this sketch as a polished image. Return only the prompt."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sketch_b64}"}}
                ]
            }],
            max_tokens=200
        )
        final_prompt = vision.choices[0].message.content.strip()
    except Exception:
        final_prompt = prompt or "a refined detailed artwork"

    # Step 2: generate from the faithful description
    response = requests.post(
        "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
        headers=HF_HEADERS,
        json={"inputs": final_prompt},
        timeout=60
    )
    if response.status_code != 200:
        return jsonify({"error": f"Generation failed: {response.text[:200]}"}), 500
    img_b64 = base64.b64encode(response.content).decode("utf-8")
    return jsonify({"url": f"data:image/jpeg;base64,{img_b64}", "prompt": final_prompt})


# ── Video generation ──
@app.route("/generate-video", methods=["POST"])
def generate_video():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt"}), 400
    try:
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/damo-vilab/text-to-video-ms-1.7b",
            headers=HF_HEADERS,
            json={"inputs": prompt},
            timeout=180
        )
        if response.status_code != 200:
            return jsonify({"error": f"Failed: {response.text[:200]}"}), 500
        video_b64 = base64.b64encode(response.content).decode("utf-8")
        return jsonify({"url": f"data:video/mp4;base64,{video_b64}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/websearch", methods=["POST"])
def websearch():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query"}), 400
    try:
        # Use DuckDuckGo instant answer API (no key needed)
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8
        )
        ddg = resp.json()
        abstract = ddg.get("AbstractText", "")
        answer = ddg.get("Answer", "")
        related = [r.get("Text", "") for r in ddg.get("RelatedTopics", [])[:3] if r.get("Text")]
        source_url = ddg.get("AbstractURL", "")

        # Build a summary via Groq
        raw = f"Abstract: {abstract}\nAnswer: {answer}\nRelated: {'; '.join(related)}"
        if not abstract and not answer:
            raw = f"No direct answer found for: {query}"

        r = get_groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"Summarize this search result for the query '{query}' in 2-3 sentences:\n{raw}"}],
            max_tokens=200
        )
        summary = r.choices[0].message.content.strip()
        return jsonify({"summary": summary, "source": source_url, "raw": raw})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Security tools ──
@app.route("/security/hash", methods=["POST"])
def security_hash():
    data = request.get_json()
    text = data.get("text", "")
    algo = data.get("algo", "md5").lower()
    algos = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    if algo not in algos:
        return jsonify({"error": f"Unsupported algorithm. Use: {', '.join(algos)}"}), 400
    h = algos[algo](text.encode()).hexdigest()
    return jsonify({"hash": h, "algo": algo})


@app.route("/security/password-strength", methods=["POST"])
def password_strength():
    data = request.get_json()
    pwd = data.get("password", "")
    score = 0
    feedback = []
    if len(pwd) >= 8: score += 1
    else: feedback.append("Use at least 8 characters")
    if len(pwd) >= 12: score += 1
    if re.search(r"[A-Z]", pwd): score += 1
    else: feedback.append("Add uppercase letters")
    if re.search(r"[a-z]", pwd): score += 1
    else: feedback.append("Add lowercase letters")
    if re.search(r"\d", pwd): score += 1
    else: feedback.append("Add numbers")
    if re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", pwd): score += 1
    else: feedback.append("Add special characters")
    levels = ["Very Weak", "Weak", "Fair", "Good", "Strong", "Very Strong"]
    return jsonify({"score": score, "max": 6, "level": levels[min(score, 5)], "feedback": feedback})


@app.route("/security/encode", methods=["POST"])
def security_encode():
    data = request.get_json()
    text = data.get("text", "")
    mode = data.get("mode", "encode")
    fmt = data.get("format", "base64")
    try:
        if fmt == "base64":
            if mode == "encode":
                result = base64.b64encode(text.encode()).decode()
            else:
                result = base64.b64decode(text.encode()).decode()
        elif fmt == "hex":
            if mode == "encode":
                result = text.encode().hex()
            else:
                result = bytes.fromhex(text).decode()
        else:
            return jsonify({"error": "Use base64 or hex"}), 400
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/security/portscan", methods=["POST"])
def port_scan():
    data = request.get_json()
    host = data.get("host", "").strip()
    ports_str = data.get("ports", "22,80,443,8080,3306,5432,6379,27017")
    if not host:
        return jsonify({"error": "No host"}), 400
    # Safety: only allow localhost and private ranges
    try:
        ip = socket.gethostbyname(host)
    except Exception:
        return jsonify({"error": "Cannot resolve host"}), 400
    allowed_prefixes = ("127.", "10.", "192.168.", "172.")
    if not any(ip.startswith(p) for p in allowed_prefixes):
        return jsonify({"error": "Port scanning is only allowed on localhost and private network ranges for safety."}), 403
    try:
        ports = [int(p.strip()) for p in ports_str.split(",") if p.strip().isdigit()]
        results = []
        for port in ports[:50]:  # cap at 50 ports
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            open_ = s.connect_ex((ip, port)) == 0
            s.close()
            results.append({"port": port, "open": open_})
        return jsonify({"host": host, "ip": ip, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Terminal ──
@app.route("/terminal", methods=["POST"])
def terminal():
    data = request.get_json()
    cmd = data.get("cmd", "").strip()
    cwd = data.get("cwd", os.path.expanduser("~"))
    if not cmd:
        return jsonify({"output": "", "error": "No command"}), 400
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15, cwd=cwd
        )
        return jsonify({
            "output": result.stdout,
            "error": result.stderr,
            "returncode": result.returncode,
            "cwd": cwd
        })
    except subprocess.TimeoutExpired:
        return jsonify({"output": "", "error": "Timed out (15s)", "returncode": -1})
    except Exception as e:
        return jsonify({"output": "", "error": str(e), "returncode": -1})



@app.route("/git/status", methods=["POST"])
def git_status():
    data = request.get_json()
    path = data.get("path", os.path.expanduser("~"))
    try:
        result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, cwd=path, timeout=5)
        branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, cwd=path, timeout=5)
        return jsonify({"status": result.stdout, "branch": branch.stdout.strip(), "error": result.stderr})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/git/diff", methods=["POST"])
def git_diff():
    data = request.get_json()
    path = data.get("path", os.path.expanduser("~"))
    try:
        result = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=path, timeout=5)
        return jsonify({"diff": result.stdout, "error": result.stderr})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/git/commit", methods=["POST"])
def git_commit():
    data = request.get_json()
    path = data.get("path", "")
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Commit message required"}), 400
    try:
        subprocess.run(["git", "add", "-A"], capture_output=True, cwd=path, timeout=5)
        result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, cwd=path, timeout=5)
        return jsonify({"output": result.stdout, "error": result.stderr})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Notes ──
NOTES_FILE = os.path.join(os.path.dirname(__file__), "notes.json")

@app.route("/notes", methods=["GET"])
def notes_get():
    try:
        if os.path.exists(NOTES_FILE):
            with open(NOTES_FILE) as f:
                return jsonify({"notes": json.load(f)})
        return jsonify({"notes": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/notes", methods=["POST"])
def notes_save():
    data = request.get_json()
    notes = data.get("notes", [])
    try:
        with open(NOTES_FILE, "w") as f:
            json.dump(notes, f, indent=2)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Tasks ──
TASKS_FILE = os.path.join(os.path.dirname(__file__), "tasks.json")

@app.route("/tasks", methods=["GET"])
def tasks_get():
    try:
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE) as f:
                return jsonify({"tasks": json.load(f)})
        return jsonify({"tasks": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tasks", methods=["POST"])
def tasks_save():
    data = request.get_json()
    tasks = data.get("tasks", [])
    try:
        with open(TASKS_FILE, "w") as f:
            json.dump(tasks, f, indent=2)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tasks/generate", methods=["POST"])
def tasks_generate():
    data = request.get_json()
    description = data.get("description", "").strip()
    if not description:
        return jsonify({"error": "No description"}), 400
    r = get_groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Generate a concise todo list (max 8 items) for this project/goal. Return ONLY a JSON array of strings, no explanation:\n{description}"}],
        max_tokens=300
    )
    raw = r.choices[0].message.content.strip()
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            tasks = json.loads(match.group())
            return jsonify({"tasks": tasks})
        except Exception:
            pass
    return jsonify({"tasks": [raw]})


def _enhance_prompt(prompt, media_type):
    # If prompt contains text/letters for logos, don't rewrite — preserve exactly
    logo_keywords = ["logo", "text", "letter", "word", "font", "typography", "label", "brand", "write", "spell"]
    if any(k in prompt.lower() for k in logo_keywords):
        return prompt + ", high quality, sharp, clean design, vector style"
    try:
        instruction = f"Rewrite this as a detailed vivid image generation prompt. Keep any specific text, words or letters EXACTLY as written. Under 100 words. Return only the prompt. Original: {prompt}"
        r = get_groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": instruction}],
            max_tokens=150
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return prompt


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

