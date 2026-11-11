import re
import copy
import json
import os
import random
import uuid
import requests
import threading
import subprocess
import urllib.parse
import csv
import time
from flask import Flask, Response, jsonify, request

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
HERE = os.path.dirname(os.path.abspath(__file__))
COMFY_DIR = os.environ.get("COMFY_DIR", "/content/ComfyUI")
WD_DIR = os.environ.get("WD_TAGGER_DIR", "/content/wd_tagger")
CLIENT_ID = uuid.uuid4().hex

JOB_PARAMS = {}

# ---------------------------------------------------------------------------
# PNG Metadata Embedder (A1111 + ComfyUI workflow)
# ---------------------------------------------------------------------------
def embed_png_metadata(img_path, meta_info, final_caption=""):
    if not os.path.isfile(img_path):
        return
    try:
        from PIL import Image, PngImagePlugin
        with Image.open(img_path) as im:
            w, h = im.size
            im_copy = im.copy()
            existing_info = dict(getattr(im, "info", {}))
            existing_exif = im.getexif()

        p = meta_info.get("params", {})
        seed = meta_info.get("seed", "")
        wf = meta_info.get("workflow", {})
        raw_prompt = p.get("prompt", "").strip()
        negative = p.get("negative", "").strip()
        prompt_text = final_caption.strip() if final_caption and final_caption.strip() else raw_prompt

        lines = [prompt_text]
        if negative:
            lines.append(f"Negative prompt: {negative}")

        param_parts = [
            f"Steps: {p.get('steps', 35)}",
            f"Sampler: {p.get('sampler', 'euler')}",
            f"Schedule type: {p.get('scheduler', 'normal')}",
            f"CFG scale: {p.get('cfg', 4.5)}",
            f"Seed: {seed}",
            f"Size: {w}x{h}",
            f"Model: {p.get('checkpoint', 'anima-base-v1.0.safetensors')}",
        ]

        for i in range(1, 7):
            l_name = p.get(f"lora{i}")
            if l_name and l_name != "None" and p.get(f"lora{i}_enabled", True):
                param_parts.append(f"LoRA {i}: {l_name} ({p.get(f'lora{i}_strength', 1.0)})")

        if p.get("input_image"):
            param_parts.append(f"Mode: img2img")
            param_parts.append(f"Denoise: {p.get('denoise', 0.65)}")

        lines.append(", ".join(param_parts))
        a1111_params = "\n".join(lines)

        def clean_latin1(text):
            if not text:
                return ""
            replacements = {
                '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
                '\u2014': '-', '\u2013': '-', '\u2026': '...', '\u00a0': ' ',
            }
            for orig, rep in replacements.items():
                text = text.replace(orig, rep)
            return text.encode('latin-1', 'replace').decode('latin-1')

        pnginfo = PngImagePlugin.PngInfo()
        for k, v in existing_info.items():
            if k not in ["parameters", "workflow", "prompt", "caption", "Description", "Comment"]:
                if isinstance(v, str):
                    pnginfo.add_text(k, clean_latin1(v))

        pnginfo.add_text("parameters", clean_latin1(a1111_params))
        pnginfo.add_text("caption", clean_latin1(prompt_text))
        pnginfo.add_text("Description", clean_latin1(prompt_text))
        pnginfo.add_text("Comment", clean_latin1(prompt_text))

        if wf:
            pnginfo.add_text("workflow", json.dumps(wf))

        im_copy.save(img_path, format="PNG", pnginfo=pnginfo, exif=existing_exif)
    except Exception as e:
        print(f"Metadata embedding note: {e}")

# ---------------------------------------------------------------------------
# Load Template Workflow
# ---------------------------------------------------------------------------
wf_path = os.path.join(HERE, "workflow_anima_api.json")
if not os.path.exists(wf_path):
    # Fallback to local default if file not yet present
    WORKFLOW = {
        "1": {"inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
        "2": {"inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "qwen_image"}, "class_type": "CLIPLoader"},
        "3": {"inputs": {"vae_name": "qwen_image_vae.safetensors"}, "class_type": "VAELoader"},
        "4": {"inputs": {"text": "", "clip": ["2", 0]}, "class_type": "CLIPTextEncode"},
        "5": {"inputs": {"text": "worst quality, low quality, bad anatomy, bad hands, missing fingers, extra digits, signature, watermark", "clip": ["2", 0]}, "class_type": "CLIPTextEncode"},
        "6": {"inputs": {"width": 1024, "height": 1024, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "7": {"inputs": {"seed": 0, "steps": 35, "cfg": 4.5, "sampler_name": "er_sde", "scheduler": "normal", "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}, "class_type": "KSampler"},
        "8": {"inputs": {"samples": ["7", 0], "vae": ["3", 0]}, "class_type": "VAEDecode"},
        "9": {"inputs": {"filename_prefix": "Anima", "images": ["8", 0]}, "class_type": "SaveImage"}
    }
else:
    with open(wf_path, encoding="utf-8") as f:
        WORKFLOW = json.load(f)

DEFAULTS = {
    "prompt": "",
    "negative": "worst quality, low quality, bad anatomy, bad hands, missing fingers, extra digits, signature, watermark",
    "steps": 35,
    "cfg": 4.5,
    "sampler": "er_sde",
    "scheduler": "normal",
    "aspect_ratio": "1:1",
    "width": 1024,
    "height": 1024,
    "checkpoint": "anima-base-v1.0.safetensors",
    "weight_dtype": "default",
}

def clamp(val, low, high, typ=int, default=0):
    try:
        v = typ(val)
        return max(low, min(high, v))
    except (TypeError, ValueError):
        return default

# Resolution Presets
RESOLUTIONS = {
    "1:1": (1024, 1024),
    "3:4": (896, 1152),
    "2:3": (832, 1216),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
    "3:2": (1216, 832),
    "16:9": (1344, 768),
    "custom": (1024, 1024)
}

# ---------------------------------------------------------------------------
# Workflow Builder
# ---------------------------------------------------------------------------
def build_workflow(p):
    wf = copy.deepcopy(WORKFLOW)
    seed = clamp(p.get("seed", -1), -1, 2**50, int, -1)
    if seed < 0:
        seed = random.randint(0, 2**50)

    # Positive & Negative Prompts
    wf["4"]["inputs"]["text"] = (p.get("prompt") or "").strip()
    wf["5"]["inputs"]["text"] = (p.get("negative") or "").strip()

    # KSampler Settings
    ks = wf["7"]["inputs"]
    ks["seed"] = seed
    ks["steps"] = clamp(p.get("steps"), 1, 150, int, DEFAULTS["steps"])
    ks["cfg"] = clamp(p.get("cfg"), 0.0, 30.0, float, DEFAULTS["cfg"])
    ks["sampler_name"] = p.get("sampler") or DEFAULTS["sampler"]
    ks["scheduler"] = p.get("scheduler") or DEFAULTS["scheduler"]

    # Resolution
    ar = p.get("aspect_ratio", "1:1")
    if ar in RESOLUTIONS and ar != "custom":
        w, h = RESOLUTIONS[ar]
    else:
        w = clamp(p.get("width", 1024), 256, 2048, int, 1024)
        h = clamp(p.get("height", 1024), 256, 2048, int, 1024)
        # Snap to multiple of 16
        w = (w // 16) * 16
        h = (h // 16) * 16

    wf["6"]["inputs"]["width"] = w
    wf["6"]["inputs"]["height"] = h

    # Checkpoint & Precision
    wf["1"]["inputs"]["unet_name"] = p.get("checkpoint") or DEFAULTS["checkpoint"]
    wf["1"]["inputs"]["weight_dtype"] = p.get("weight_dtype") or DEFAULTS.get("weight_dtype", "default")

    # Chain up to 6 LoRAs dynamically
    curr_model = ["1", 0]
    curr_clip = ["2", 0]

    for i in range(1, 7):
        l_name = p.get(f"lora{i}", "None")
        l_enabled = p.get(f"lora{i}_enabled", True)
        if isinstance(l_enabled, str):
            l_enabled = l_enabled.lower() in ("true", "1", "yes")
        try:
            l_str = float(p.get(f"lora{i}_strength", 1.0))
        except (ValueError, TypeError):
            l_str = 1.0
        # Allow wide range from -10.0 to 10.0 for extreme LoRA styling
        l_str = max(-10.0, min(10.0, l_str))

        if l_name and l_name != "None" and l_enabled and l_str != 0:
            node_id = f"lora_{i}"
            wf[node_id] = {
                "inputs": {
                    "lora_name": l_name,
                    "strength_model": l_str,
                    "strength_clip": l_str,
                    "model": curr_model,
                    "clip": curr_clip
                },
                "class_type": "LoraLoader",
                "_meta": {"title": f"LoRA {i} ({l_name})"}
            }
            curr_model = [node_id, 0]
            curr_clip = [node_id, 1]

    # Connect model to KSampler
    ks["model"] = curr_model

    # Connect clip to TextEncoders
    wf["4"]["inputs"]["clip"] = curr_clip
    wf["5"]["inputs"]["clip"] = curr_clip

    # Tab routing: Text-to-Image vs Img2Img
    active_tab = p.get("tab", "gen")
    input_img = (p.get("input_image") or "").strip()

    if active_tab == "img2img" and input_img:
        # Guarantee input image exists in ComfyUI/input folder
        in_dir = os.path.join(COMFY_DIR, "input")
        os.makedirs(in_dir, exist_ok=True)
        in_path = os.path.join(in_dir, input_img)
        if not os.path.isfile(in_path):
            out_dir = os.path.join(COMFY_DIR, "output")
            if os.path.isdir(out_dir):
                for root, _, files in os.walk(out_dir):
                    if input_img in files:
                        import shutil
                        shutil.copy2(os.path.join(root, input_img), in_path)
                        break

        resize_mode = p.get("img2img_resize", "crop")

        # Determine target dimensions & crop behavior
        tw, th = w, h
        crop_method = "center"

        if resize_mode == "original" and os.path.isfile(in_path):
            try:
                from PIL import Image
                with Image.open(in_path) as im:
                    ow, oh = im.size
                scale = min(1.0, 1024.0 / max(ow, oh))
                tw = max(256, (int(ow * scale) // 16) * 16)
                th = max(256, (int(oh * scale) // 16) * 16)
                crop_method = "disabled"
            except Exception:
                tw, th = w, h
                crop_method = "center"
        elif resize_mode == "stretch":
            tw, th = w, h
            crop_method = "disabled"
        else: # "crop" (default): follow selected aspect ratio with center crop
            tw, th = w, h
            crop_method = "center"

        denoise_val = clamp(p.get("denoise", 0.65), 0.05, 1.0, float, 0.65)
        wf["input_load"] = {
            "inputs": {"image": input_img},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Input Image"}
        }
        wf["input_scale"] = {
            "inputs": {
                "image": ["input_load", 0],
                "upscale_method": "bicubic",
                "width": tw,
                "height": th,
                "crop": crop_method
            },
            "class_type": "ImageScale",
            "_meta": {"title": "Scale Input Image"}
        }
        wf["input_encode"] = {
            "inputs": {
                "pixels": ["input_scale", 0],
                "vae": ["3", 0]
            },
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"}
        }
        ks["latent_image"] = ["input_encode", 0]
        ks["denoise"] = denoise_val
    else:
        ks["latent_image"] = ["6", 0]
        ks["denoise"] = 1.0

    return wf, seed

def explain_validation(resp):
    try:
        data = resp.json()
    except Exception:
        return f"Backend rejected the request ({resp.status_code})."
    parts = []
    err = data.get("error")
    if isinstance(err, dict):
        parts.append(err.get("message", "Invalid workflow"))
        if err.get("details"):
            parts.append(str(err["details"]))
    for nid, info in (data.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            parts.append(f'{info.get("class_type", nid)}: {e.get("message", "")} {e.get("details", "")}'.strip())
    return " | ".join(parts) or "Backend rejected the workflow."

# ---------------------------------------------------------------------------
# SmilingWolf/wd-eva02-large-tagger-v3 Implementation
# ---------------------------------------------------------------------------
_wd_session = None
_wd_tags_info = None
_wd_lock = threading.Lock()

def get_wd_tagger():
    global _wd_session, _wd_tags_info
    with _wd_lock:
        if _wd_session is None:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            os.makedirs(WD_DIR, exist_ok=True)
            model_file = os.path.join(WD_DIR, "model.onnx")
            tags_file = os.path.join(WD_DIR, "selected_tags.csv")

            if not os.path.exists(model_file):
                print("Downloading SmilingWolf/wd-eva02-large-tagger-v3 (model.onnx)...")
                hf_hub_download(
                    repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
                    filename="model.onnx",
                    local_dir=WD_DIR
                )
            if not os.path.exists(tags_file):
                print("Downloading SmilingWolf/wd-eva02-large-tagger-v3 (selected_tags.csv)...")
                hf_hub_download(
                    repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
                    filename="selected_tags.csv",
                    local_dir=WD_DIR
                )

            # Load tag list
            tags_list = []
            with open(tags_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        tag_name = row[1].replace("_", " ")
                        cat = int(row[2])
                        tags_list.append((tag_name, cat))
            _wd_tags_info = tags_list

            # ONNX Session
            providers = ["CPUExecutionProvider"]
            print(f"Loading WD-EVA02 ONNX session on CPU to preserve 100% GPU VRAM for Anima...")
            _wd_session = ort.InferenceSession(model_file, providers=providers)
            print("✓ WD-EVA02 Tagger loaded successfully.")
    return _wd_session, _wd_tags_info

def run_wd_tagger(image_path, general_threshold=0.35, character_threshold=0.75):
    from PIL import Image
    import numpy as np

    session, tags_info = get_wd_tagger()
    input_name = session.get_inputs()[0].name

    with Image.open(image_path) as raw_img:
        if raw_img.mode in ("RGBA", "LA") or (raw_img.mode == "P" and "transparency" in raw_img.info):
            canvas = Image.new("RGBA", raw_img.size, (255, 255, 255))
            canvas.alpha_composite(raw_img.convert("RGBA"))
            img = canvas.convert("RGB")
        else:
            img = raw_img.convert("RGB")

    # Pad to square with white background
    w, h = img.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(img, (pad_left, pad_top))

    # Resize to 448x448
    target_size = 448
    if max_dim != target_size:
        padded = padded.resize((target_size, target_size), Image.Resampling.BICUBIC)

    # Convert to float32 BGR [1, 448, 448, 3] unnormalized (0-255)
    arr = np.array(padded, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB to BGR
    arr = np.expand_dims(arr, axis=0)

    # Run inference
    outputs = session.run(None, {input_name: arr})
    scores = outputs[0][0]

    characters = []
    artists = []
    general = []
    ratings = {}

    for (tag_name, cat), score in zip(tags_info, scores):
        s = float(score)
        if cat == 9:
            ratings[tag_name] = round(s, 3)
        elif cat == 4:
            if s >= character_threshold:
                characters.append((tag_name, round(s, 3)))
        elif cat == 1:
            if s >= character_threshold:
                artists.append((f"@{tag_name}", round(s, 3)))
        elif cat == 0:
            if s >= general_threshold:
                general.append((tag_name, round(s, 3)))

    characters.sort(key=lambda x: x[1], reverse=True)
    artists.sort(key=lambda x: x[1], reverse=True)
    general.sort(key=lambda x: x[1], reverse=True)

    # Anima preferred prompt structure: Characters, Artists (@name), General tags
    tokens = [c[0] for c in characters] + [a[0] for a in artists] + [g[0] for g in general]
    caption = ", ".join(tokens)

    return {
        "caption": caption,
        "characters": characters,
        "artists": artists,
        "general": general,
        "ratings": ratings
    }

# ---------------------------------------------------------------------------
# Booru Tag Autocomplete Engine
# ---------------------------------------------------------------------------
_booru_tags = None

def load_booru_tags():
    global _booru_tags
    if _booru_tags is not None:
        return _booru_tags

    tags = []
    danbooru_file = os.path.join(WD_DIR, "danbooru.csv")
    selected_file = os.path.join(WD_DIR, "selected_tags.csv")

    if os.path.exists(danbooru_file):
        try:
            with open(danbooru_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            t_name = row[0].replace("_", " ")
                            t_cat = int(row[1])
                            t_count = int(row[2])
                            tags.append((t_name, t_cat, t_count))
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Error reading danbooru.csv: {e}")

    if not tags and os.path.exists(selected_file):
        try:
            with open(selected_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 4:
                        try:
                            t_name = row[1].replace("_", " ")
                            t_cat = int(row[2])
                            t_count = int(row[3]) if row[3].isdigit() else 0
                            tags.append((t_name, t_cat, t_count))
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Error reading selected_tags.csv: {e}")

    if not tags:
        # High-frequency Danbooru starter tags for immediate autocomplete fallback
        starter = [
            ("1girl", 0, 5200000), ("solo", 0, 4800000), ("long hair", 0, 2400000),
            ("looking at viewer", 0, 2300000), ("smile", 0, 2100000), ("short hair", 0, 1900000),
            ("breasts", 0, 1800000), ("blush", 0, 1700000), ("blue eyes", 0, 1500000),
            ("open mouth", 0, 1400000), ("simple background", 0, 1300000), ("white background", 0, 1200000),
            ("black hair", 0, 1100000), ("large breasts", 0, 1050000), ("brown hair", 0, 1000000),
            ("twintails", 0, 850000), ("ponytail", 0, 800000), ("skirt", 0, 950000),
            ("school uniform", 0, 900000), ("thighhighs", 0, 850000), ("dress", 0, 820000),
            ("gloves", 0, 800000), ("hair ornament", 0, 780000), ("collarbone", 0, 750000),
            ("cleavage", 0, 720000), ("navel", 0, 700000), ("bare shoulders", 0, 680000),
            ("ribbon", 0, 650000), ("bow", 0, 640000), ("jacket", 0, 620000),
            ("shirt", 0, 600000), ("underwear", 0, 580000), ("panties", 0, 550000),
            ("bra", 0, 520000), ("lingerie", 0, 490000), ("garter belt", 0, 420000),
            ("sitting", 0, 610000), ("standing", 0, 590000), ("lying", 0, 390000),
            ("masterpiece", 0, 2900000), ("best quality", 0, 2800000), ("score 9", 0, 1500000),
            ("sensitive", 0, 1200000), ("newest", 0, 900000),
            ("hatsune miku", 4, 150000), ("ouro kronii", 4, 45000), ("gawr gura", 4, 60000),
            ("frieren", 4, 55000), ("rem (re:zero)", 4, 85000), ("asuka langley soryu", 4, 90000),
            ("kamado nezuko", 4, 70000), ("makima (chainsaw man)", 4, 48000), ("2b (nier:automata)", 4, 65000),
            ("tifa lockhart", 4, 58000),
            ("taiyang yu", 1, 10000), ("bboky0", 1, 8000), ("tony taka", 1, 15000),
            ("kantoku", 1, 18000), ("citemori", 1, 5000), ("sho (sho_lwlw)", 1, 7000),
            ("hiten", 1, 12000), ("anmi", 1, 14000), ("tiv", 1, 11000), ("wlop", 1, 20000)
        ]
        tags = starter

    _booru_tags = tags
    return _booru_tags

# ---------------------------------------------------------------------------
# Robust Background Model & LoRA Downloader (Civitai, HuggingFace, GDrive)
# ---------------------------------------------------------------------------
VALID_EXTS = (".safetensors", ".pt", ".bin", ".ckpt")

def ensure_valid_ext(fname, default_ext=".safetensors"):
    fname = re.sub(r'[<>:"/\\|?*]', "_", fname.strip())
    if not fname.lower().endswith(VALID_EXTS):
        fname += default_ext
    return fname

def filename_from_headers(headers, final_url=""):
    cd = headers.get("Content-Disposition", "")
    if cd:
        m = re.search(r"filename\*=UTF-8''([^;]+)", cd, re.I)
        if m:
            return urllib.parse.unquote(m.group(1)).strip().strip('"').strip("'")
        m = re.search(r'filename="?([^";]+)"?', cd, re.I)
        if m:
            return urllib.parse.unquote(m.group(1)).strip().strip('"').strip("'")
    if final_url:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(final_url).query)
        rcd = qs.get("response-content-disposition", [""])[0]
        if rcd:
            m = re.search(r'filename="?([^";]+)"?', urllib.parse.unquote(rcd), re.I)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    return None

DOWNLOAD_STATE = {
    "active": False,
    "status": "idle",
    "percent": 0,
    "msg": "",
    "error": None,
    "filename": "",
    "target_type": ""
}
DOWNLOAD_LOCK = threading.Lock()

def download_file_robust(final_url, dest_dir, fname, headers=None):
    global DOWNLOAD_STATE
    UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    
    # Check if final_url is an AWS S3 / Cloudflare R2 presigned URL
    is_presigned = any(k in final_url for k in ["X-Amz-Signature", "X-Amz-Algorithm", "r2.cloudflarestorage.com", "amazonaws.com"])
    send_auth = bool(headers and headers.get("Authorization") and not is_presigned)

    cmd = [
        "aria2c", "-x", "16", "-s", "16", "-k", "1M", "-c", "--file-allocation=none",
        "--console-log-level=warn", "--summary-interval=1",
        f"--user-agent={UA}",
        "--check-certificate=false",
        "-d", dest_dir, "-o", fname, final_url
    ]
    if not is_presigned:
        cmd.append("--header=Referer: https://civitai.com/")
    if send_auth:
        cmd.extend(["--header", f"Authorization: {headers['Authorization']}"])

    err_lines = []
    aria_ok = False
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            line_str = line.strip()
            if any(k in line_str for k in ["ERROR", "errorCode", "Exception", "failed"]):
                err_lines.append(line_str)
            m_pct = re.search(r'\((\d+)%\)', line_str)
            if m_pct:
                DOWNLOAD_STATE["percent"] = int(m_pct.group(1))
            m_prog = re.search(r'(\d+(?:\.\d+)?[A-Za-z]+/\d+(?:\.\d+)?[A-Za-z]+\(\d+%\))', line_str)
            m_spd = re.search(r'DL:([^\s]+)', line_str)
            if m_prog:
                DOWNLOAD_STATE["msg"] = f"Downloading: {m_prog.group(1)}" + (f" • {m_spd.group(1)}/s" if m_spd else "")
            elif line_str:
                DOWNLOAD_STATE["msg"] = line_str[:100]
        proc.wait()
        if proc.returncode == 0:
            aria_ok = True
        else:
            print(f"aria2c returned code {proc.returncode}, fallback to direct streaming: {err_lines}")
    except Exception as e:
        print(f"aria2c execution error ({e}), fallback to direct streaming...")

    if not aria_ok:
        out_path = os.path.join(dest_dir, fname)
        DOWNLOAD_STATE["msg"] = f"Streaming {fname} via direct connection..."
        req_headers = {"User-Agent": UA}
        if not is_presigned:
            req_headers["Referer"] = "https://civitai.com/"
        if send_auth:
            req_headers["Authorization"] = headers["Authorization"]

        with requests.get(final_url, headers=req_headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            total_len = int(r.headers.get("content-length", 0))
            dl_bytes = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        dl_bytes += len(chunk)
                        if total_len > 0:
                            pct = int(dl_bytes / total_len * 100)
                            DOWNLOAD_STATE["percent"] = pct
                            DOWNLOAD_STATE["msg"] = f"Downloading: {dl_bytes // (1024*1024)}MB / {total_len // (1024*1024)}MB ({pct}%)"
                        else:
                            DOWNLOAD_STATE["msg"] = f"Downloading: {dl_bytes // (1024*1024)}MB"

def run_download_task(raw_entry, target_type, civitai_key, custom_name_override=""):
    global DOWNLOAD_STATE
    try:
        url = raw_entry.strip()
        custom_name = custom_name_override.strip() if custom_name_override else None
        if "|" in url and not url.startswith(("http://", "https://", "urn:")):
            custom_name, url = [p.strip() for p in url.split("|", 1)]

        m_air = re.match(r"urn:air:(?P<eco>[^:]+):(?P<type>[^:]+):(?P<src>[^:]+):(?P<model_id>\d+)(?:@(?P<version_id>\d+))?(?:\+(?P<file_id>\d+))?", url)
        if m_air:
            air_type = m_air.group("type").lower()
            if "checkpoint" in air_type or "model" in air_type or "diffusion" in air_type:
                target_type = "checkpoint"
            elif "lora" in air_type:
                target_type = "lora"

            vid = m_air.group("version_id")
            fid = m_air.group("file_id")
            mid = m_air.group("model_id")

            if vid:
                url = f"https://civitai.com/api/download/models/{vid}"
                if fid:
                    url += f"?fileId={fid}"
            else:
                url = f"https://civitai.com/models/{mid}"

        domain = urllib.parse.urlparse(url).netloc.lower()
        is_presigned = any(k in url for k in ["X-Amz-Signature", "X-Amz-Algorithm", "r2.cloudflarestorage.com", "amazonaws.com"])

        if target_type == "auto":
            if is_presigned:
                test_str = (url + " " + (custom_name or "")).lower()
                if any(k in test_str for k in ["checkpoint", "diffusion_models", "diffusion_model"]):
                    target_type = "checkpoint"
                else:
                    target_type = "lora"
            elif "civitai" in domain:
                parsed = urllib.parse.urlparse(url)
                m_page = re.search(r"/models/(\d+)", parsed.path)
                m_dl = re.search(r"/api/download/models/(\d+)", parsed.path)
                qs = urllib.parse.parse_qs(parsed.query)
                headers = {"User-Agent": "Mozilla/5.0"}
                if civitai_key:
                    headers["Authorization"] = f"Bearer {civitai_key}"
                try:
                    if m_page:
                        mid = m_page.group(1)
                        r = requests.get(f"https://civitai.com/api/v1/models/{mid}", headers=headers, timeout=15)
                        if r.status_code == 200:
                            m_type = r.json().get("type", "").lower()
                            target_type = "checkpoint" if "checkpoint" in m_type else "lora"
                    elif m_dl or "modelVersionId" in qs:
                        vid = m_dl.group(1) if m_dl else qs["modelVersionId"][0]
                        r = requests.get(f"https://civitai.com/api/v1/model-versions/{vid}", headers=headers, timeout=15)
                        if r.status_code == 200:
                            m_type = r.json().get("model", {}).get("type", "").lower()
                            target_type = "checkpoint" if "checkpoint" in m_type else "lora"
                except Exception:
                    pass
            if target_type == "auto":
                if any(k in url.lower() for k in ["checkpoint", "diffusion_models", "diffusion"]):
                    target_type = "checkpoint"
                else:
                    target_type = "lora"

        dest_dir = os.path.join(COMFY_DIR, "models", "diffusion_models" if target_type == "checkpoint" else "loras")
        os.makedirs(dest_dir, exist_ok=True)
        label = "Diffusion Model" if target_type == "checkpoint" else "LoRA"

        DOWNLOAD_STATE["target_type"] = target_type
        DOWNLOAD_STATE["msg"] = f"Resolving download link for {label}..."
        DOWNLOAD_STATE["percent"] = 5

        if "drive.google.com" in domain or "docs.google.com" in domain:
            DOWNLOAD_STATE["msg"] = f"Downloading {label} from Google Drive..."
            try:
                import gdown
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "gdown"], check=True)
                import gdown

            if "/folders/" in url:
                gdown.download_folder(url=url, output=dest_dir, quiet=False, remaining_ok=True)
                fname = "Google Drive Folder"
            else:
                out_target = os.path.join(dest_dir, ensure_valid_ext(custom_name)) if custom_name else (dest_dir + "/")
                out_path = gdown.download(url=url, output=out_target, quiet=False, fuzzy=True)
                if not out_path or not os.path.exists(out_path):
                    raise RuntimeError("Google Drive download failed. Ensure sharing is set to 'Anyone with the link'.")
                if not out_path.lower().endswith(VALID_EXTS):
                    new_path = out_path + ".safetensors"
                    os.replace(out_path, new_path)
                    out_path = new_path
                fname = os.path.basename(out_path)

        elif is_presigned:
            DOWNLOAD_STATE["msg"] = f"Downloading {label} from direct storage link..."
            fname = custom_name or filename_from_headers({}, url) or urllib.parse.unquote(url.split("?")[0].rstrip("/").split("/")[-1])
            fname = ensure_valid_ext(fname)
            DOWNLOAD_STATE["filename"] = fname
            download_file_robust(url, dest_dir, fname, headers=None)

        elif "civitai" in domain:
            DOWNLOAD_STATE["msg"] = "Connecting to Civitai API..."
            parsed = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed.query)
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            if civitai_key:
                headers["Authorization"] = f"Bearer {civitai_key}"

            version_id = None
            dl_url = url
            fname = custom_name

            m_page = re.search(r"/models/(\d+)", parsed.path)
            m_dl = re.search(r"/api/download/models/(\d+)", parsed.path)

            if m_dl:
                version_id = m_dl.group(1)
            elif m_page:
                if "modelVersionId" in qs:
                    version_id = qs["modelVersionId"][0]
                else:
                    mid = m_page.group(1)
                    r = requests.get(f"https://civitai.com/api/v1/models/{mid}", headers=headers, timeout=30)
                    r.raise_for_status()
                    versions = r.json().get("modelVersions") or []
                    if not versions:
                        raise RuntimeError(f"No model versions found on Civitai for model {mid}")
                    version_id = str(versions[0]["id"])
                dl_url = f"https://civitai.com/api/download/models/{version_id}"

            if version_id and not fname:
                try:
                    r_ver = requests.get(f"https://civitai.com/api/v1/model-versions/{version_id}", headers=headers, timeout=30)
                    if r_ver.status_code == 200:
                        vdata = r_ver.json()
                        files = vdata.get("files") or []
                        chosen = next((f for f in files if f.get("primary")), files[0] if files else None)
                        if chosen:
                            fname = chosen.get("name")
                            if chosen.get("downloadUrl"):
                                dl_url = chosen["downloadUrl"]
                except Exception:
                    pass

            if civitai_key and "token=" not in dl_url:
                sep = "&" if "?" in dl_url else "?"
                dl_url = f"{dl_url}{sep}token={urllib.parse.quote(civitai_key)}"

            resp = requests.get(dl_url, headers=headers, allow_redirects=False, timeout=30)
            if resp.status_code in (301, 302, 303, 307, 308) and "Location" in resp.headers:
                final_url = resp.headers["Location"]
                if "civitai.com/login" in final_url:
                    raise RuntimeError("Civitai requires an API Key for this model. Please enter your Civitai API Key.")
                if not fname:
                    fname = filename_from_headers(resp.headers, final_url)
            elif resp.status_code in (401, 403):
                raise RuntimeError(f"Civitai returned HTTP {resp.status_code}. Please provide a valid Civitai API Key.")
            else:
                resp.raise_for_status()
                final_url = dl_url
                if not fname:
                    fname = filename_from_headers(resp.headers, final_url)

            if not fname:
                fname = f"civitai_{label.lower().replace(' ', '_')}_{version_id or 'custom'}.safetensors"
            fname = ensure_valid_ext(fname)

            DOWNLOAD_STATE["filename"] = fname
            DOWNLOAD_STATE["msg"] = f"Downloading {fname}..."
            download_file_robust(final_url, dest_dir, fname, headers=None)

        else:
            if "huggingface.co" in domain and "/blob/" in url:
                url = url.replace("/blob/", "/resolve/", 1)
            fname = custom_name or urllib.parse.unquote(url.split("?")[0].rstrip("/").split("/")[-1])
            fname = ensure_valid_ext(fname)
            DOWNLOAD_STATE["filename"] = fname
            DOWNLOAD_STATE["msg"] = f"Downloading {fname}..."
            download_file_robust(url, dest_dir, fname)

        DOWNLOAD_STATE["percent"] = 100
        DOWNLOAD_STATE["msg"] = f"Successfully downloaded {DOWNLOAD_STATE.get('filename')}!"
        DOWNLOAD_STATE["status"] = "done"

    except Exception as e:
        import traceback
        traceback.print_exc()
        DOWNLOAD_STATE["status"] = "error"
        DOWNLOAD_STATE["error"] = str(e)
        DOWNLOAD_STATE["msg"] = f"Download failed: {str(e)}"
    finally:
        DOWNLOAD_STATE["active"] = False

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
PROGRESS_STATE = {"value": 0, "max": 0, "percent": 0, "node": ""}

def init_comfy_ws_listener():
    def listener():
        global PROGRESS_STATE
        try:
            import websocket
        except ImportError:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "websocket-client"], check=False)
                import websocket
            except Exception:
                return

        ws_host = COMFY.replace("http://", "").replace("https://", "").rstrip("/")
        ws_url = f"ws://{ws_host}/ws?clientId={CLIENT_ID}"

        while True:
            try:
                ws = websocket.create_connection(ws_url, timeout=10)
                while True:
                    msg = ws.recv()
                    if not msg:
                        break
                    if isinstance(msg, str):
                        try:
                            data = json.loads(msg)
                            m_type = data.get("type")
                            if m_type == "progress":
                                p_data = data.get("data", {})
                                val = p_data.get("value", 0)
                                m = p_data.get("max", 0)
                                pct = int(val / m * 100) if m > 0 else 0
                                PROGRESS_STATE["value"] = val
                                PROGRESS_STATE["max"] = m
                                PROGRESS_STATE["percent"] = pct
                            elif m_type == "executing":
                                node = data.get("data", {}).get("node")
                                PROGRESS_STATE["node"] = node or ""
                                if not node:
                                    PROGRESS_STATE["percent"] = 100
                        except Exception:
                            pass
            except Exception:
                time.sleep(2)

    t = threading.Thread(target=listener, daemon=True)
    t.start()

init_comfy_ws_listener()

app = Flask(__name__)

def comfy_get(path):
    r = requests.get(f"{COMFY}{path}", timeout=10)
    r.raise_for_status()
    return r.json()

@app.get("/api/models")
def get_models():
    diff_dir = os.path.join(COMFY_DIR, "models", "diffusion_models")
    lora_dir = os.path.join(COMFY_DIR, "models", "loras")
    
    models = []
    if os.path.isdir(diff_dir):
        for f in os.listdir(diff_dir):
            if f.lower().endswith((".safetensors", ".ckpt", ".pt", ".bin")):
                models.append(f)
    if not models:
        models = ["anima-base-v1.0.safetensors"]

    loras = ["None"]
    if os.path.isdir(lora_dir):
        for f in sorted(os.listdir(lora_dir)):
            if f.lower().endswith((".safetensors", ".pt")):
                loras.append(f)

    samplers = ["er_sde", "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_3m_sde", "dpmpp_sde_gpu", "ddim", "uni_pc", "heun", "lms"]
    schedulers = ["normal", "simple", "karras", "sgm_uniform", "exponential"]
    try:
        k_info = comfy_get("/object_info/KSampler")
        req_inputs = k_info.get("KSampler", {}).get("input", {}).get("required", {})
        avail_samplers = req_inputs.get("sampler_name", [[]])[0]
        if avail_samplers:
            s_list = []
            if "er_sde" in avail_samplers:
                s_list.append("er_sde")
            for s in avail_samplers:
                if s not in s_list:
                    s_list.append(s)
            samplers = s_list
        avail_sched = req_inputs.get("scheduler", [[]])[0]
        if avail_sched:
            schedulers = avail_sched
    except Exception:
        pass

    return jsonify(
        models=models,
        loras=loras,
        samplers=samplers,
        schedulers=schedulers,
        resolutions=RESOLUTIONS
    )

@app.get("/api/tags")
def search_tags():
    q = (request.args.get("q") or "").strip().lower().replace("_", " ")
    tags = load_booru_tags()
    if not q:
        return jsonify(tags=[{"name": t[0], "cat": t[1], "count": t[2]} for t in tags[:20]])

    is_artist_q = q.startswith("@")
    clean_q = q[1:] if is_artist_q else q

    matches = []
    # 1) Prefix match
    for name, cat, count in tags:
        comp_name = f"@{name}" if cat == 1 else name
        if comp_name.lower().startswith(q) or name.lower().startswith(clean_q):
            matches.append({"name": comp_name, "cat": cat, "count": count})
            if len(matches) >= 15:
                break

    # 2) Substring match if less than 15
    if len(matches) < 15:
        for name, cat, count in tags:
            comp_name = f"@{name}" if cat == 1 else name
            if clean_q in name.lower() and not any(m["name"] == comp_name for m in matches):
                matches.append({"name": comp_name, "cat": cat, "count": count})
                if len(matches) >= 15:
                    break

    return jsonify(tags=matches)

@app.post("/api/generate")
def generate():
    try:
        p = request.get_json(force=True, silent=True) or {}
        if not (p.get("prompt") or "").strip():
            return jsonify(error="Please enter a prompt."), 400
        wf, seed = build_workflow(p)
    except Exception as e:
        return jsonify(error=f"Workflow build error: {str(e)}"), 500

    try:
        # Soft empty cache to prevent VRAM memory leaks between runs
        try:
            requests.post(f"{COMFY}/free", json={"unload_models": False, "free_memory": True}, timeout=2)
        except Exception:
            pass

        r = requests.post(
            f"{COMFY}/prompt",
            json={
                "prompt": wf,
                "client_id": CLIENT_ID,
                "extra_data": {"extra_pnginfo": {"workflow": wf}}
            },
            timeout=30
        )
    except requests.RequestException as e:
        return jsonify(error=f"ComfyUI backend not reachable: {e}"), 502

    if r.status_code != 200:
        return jsonify(error=explain_validation(r)), 400

    resp_data = r.json()
    prompt_id = resp_data.get("prompt_id")
    JOB_PARAMS[prompt_id] = {"params": p, "seed": seed, "workflow": wf}
    return jsonify(prompt_id=prompt_id, seed=seed)

@app.post("/api/cancel")
def cancel_generation():
    global PROGRESS_STATE
    try:
        requests.post(f"{COMFY}/interrupt", timeout=5)
        PROGRESS_STATE = {"value": 0, "max": 0, "percent": 0, "node": ""}
        return jsonify(status="interrupted")
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.get("/api/status/<pid>")
def check_status(pid):
    try:
        hist = comfy_get(f"/history/{pid}")
    except Exception as e:
        return jsonify(state="error", error=f"Backend query failed: {e}")

    if pid in hist:
        entry = hist[pid]
        outputs = entry.get("outputs", {})
        images = []
        for nid, out in outputs.items():
            for img in out.get("images", []):
                images.append({
                    "filename": img.get("filename"),
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output")
                })
        if images:
            meta_info = JOB_PARAMS.get(pid, {})
            seed = meta_info.get("seed", "")
            final_img = images[-1]
            out_fn = final_img["filename"]
            local_path = os.path.join(COMFY_DIR, "output", final_img.get("subfolder", ""), out_fn)
            embed_png_metadata(local_path, meta_info)
            return jsonify(
                state="done",
                images=images,
                seed=seed,
                download_name=f"{os.path.splitext(out_fn)[0]}_seed{seed}.png"
            )
        err = entry.get("status", {}).get("messages", [])
        return jsonify(state="error", error=f"Completed with no image: {err}")

    try:
        q = comfy_get("/queue")
        running = [item[1] for item in q.get("queue_running", [])]
        pending = [item[1] for item in q.get("queue_pending", [])]
        if pid in running:
            return jsonify(state="running", progress=PROGRESS_STATE)
        if pid in pending:
            return jsonify(state="queued", position=pending.index(pid) + 1, progress=PROGRESS_STATE)
    except Exception:
        pass
    return jsonify(state="queued", progress=PROGRESS_STATE)

@app.get("/api/view")
def view_image():
    fn = request.args.get("filename")
    sub = request.args.get("subfolder", "")
    t = request.args.get("type", "output")
    folder = os.path.join(COMFY_DIR, t, sub)
    full_path = os.path.normpath(os.path.join(folder, fn))
    if not full_path.startswith(folder) or not os.path.isfile(full_path):
        return ("Not found", 404)
    ext = os.path.splitext(fn)[1].lower()
    mimes = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    return Response(open(full_path, "rb").read(), mimetype=mimes.get(ext, "image/png"))

@app.post("/api/copy_to_input")
def copy_to_input():
    p = request.get_json(force=True, silent=True) or {}
    fn = p.get("filename", "").strip()
    if not fn:
        return jsonify(error="No filename provided"), 400

    in_dir = os.path.join(COMFY_DIR, "input")
    os.makedirs(in_dir, exist_ok=True)
    dest_path = os.path.join(in_dir, fn)

    if os.path.isfile(dest_path):
        return jsonify(filename=fn)

    out_dir = os.path.join(COMFY_DIR, "output")
    if os.path.isdir(out_dir):
        for root, _, files in os.walk(out_dir):
            if fn in files:
                import shutil
                shutil.copy2(os.path.join(root, fn), dest_path)
                return jsonify(filename=fn)

    return jsonify(filename=fn)

@app.post("/api/upload")
def upload_file():
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify(error="Empty filename"), 400
    dest_dir = os.path.join(COMFY_DIR, "input")
    os.makedirs(dest_dir, exist_ok=True)
    ext = os.path.splitext(f.filename)[1].lower() or ".png"
    safe_name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    out_path = os.path.join(dest_dir, safe_name)
    f.save(out_path)
    return jsonify(filename=safe_name)

@app.post("/api/tag_image")
def tag_image_endpoint():
    p = request.get_json(force=True, silent=True) or {}
    fn = p.get("filename", "").strip()
    if not fn:
        return jsonify(error="No image specified"), 400
    img_path = os.path.join(COMFY_DIR, "input", fn)
    if not os.path.isfile(img_path):
        return jsonify(error="Image file not found on server"), 404

    gen_th = clamp(p.get("general_threshold", 0.35), 0.05, 0.95, float, 0.35)
    char_th = clamp(p.get("character_threshold", 0.75), 0.10, 0.99, float, 0.75)

    try:
        result = run_wd_tagger(img_path, general_threshold=gen_th, character_threshold=char_th)
        return jsonify(result)
    except Exception as e:
        return jsonify(error=f"WD-EVA02 error: {str(e)}"), 500

@app.post("/api/download")
def download_model():
    global DOWNLOAD_STATE
    p = request.get_json(force=True, silent=True) or {}
    url = (p.get("url") or "").strip()
    target_type = (p.get("type") or p.get("folder") or "auto").strip().lower()
    civitai_key = (p.get("api_key") or p.get("civitai_key") or "").strip()
    custom_name = (p.get("filename") or "").strip()

    if not url:
        return jsonify(error="Please enter a valid URL or Civitai AIR identifier."), 400

    with DOWNLOAD_LOCK:
        if DOWNLOAD_STATE.get("active"):
            return jsonify(error="Another download is already in progress. Please wait for it to finish."), 409
        DOWNLOAD_STATE = {
            "active": True,
            "status": "downloading",
            "percent": 0,
            "msg": "Starting download...",
            "error": None,
            "filename": "",
            "target_type": target_type
        }

    t = threading.Thread(target=run_download_task, args=(url, target_type, civitai_key, custom_name), daemon=True)
    t.start()
    return jsonify(status="started")

@app.get("/api/download_status")
@app.get("/api/download_status/<tid>")
def download_status(tid=None):
    return jsonify(DOWNLOAD_STATE)

@app.post("/api/clear_cache")
def clear_cache():
    try:
        requests.post(f"{COMFY}/free", json={"unload_models": True, "free_memory": True}, timeout=5)
        return jsonify(status="ok")
    except Exception as e:
        return jsonify(error=str(e)), 500

# ---------------------------------------------------------------------------
# Frontend Single-Page App (Small Compact Anime UI)
# ---------------------------------------------------------------------------
INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>✨ ANIMA WebUI • Zero-Scroll Anime Generation</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090c10;
      --bg-panel: #0d1117;
      --bg-card: #161b22;
      --bg-hover: #1f242c;
      --border: #30363d;
      --accent: #a855f7;
      --accent-glow: rgba(168, 85, 247, 0.25);
      --accent-2: #ec4899;
      --accent-cyan: #06b6d4;
      --text: #f0f6fc;
      --text-muted: #8b949e;
      --danger: #f43f5e;
      --success: #10b981;
      --radius: 8px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    header {
      height: 48px;
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      flex-shrink: 0;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.3px;
    }
    .brand-badge {
      font-size: 10px;
      padding: 2px 7px;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #fff;
      font-weight: 600;
    }
    .header-controls {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .preset-pill {
      display: flex;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 2px;
      font-size: 11px;
    }
    .preset-pill button {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 4px 10px;
      border-radius: 16px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.15s;
    }
    .preset-pill button.active {
      background: var(--accent);
      color: #fff;
    }
    .app-body {
      display: flex;
      flex: 1;
      height: calc(100vh - 48px);
      overflow: hidden;
    }
    .controls-pane {
      width: 44%;
      max-width: 620px;
      min-width: 380px;
      background: var(--bg-panel);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .tabs-bar {
      display: flex;
      border-bottom: 1px solid var(--border);
      background: var(--bg-card);
    }
    .tab-btn {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 12px;
      font-weight: 600;
      padding: 10px 4px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.15s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 5px;
    }
    .tab-btn.active {
      color: #fff;
      border-bottom-color: var(--accent);
      background: rgba(168, 85, 247, 0.08);
    }
    .controls-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 14px 16px 32px 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .prompt-box {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .box-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
    }
    textarea {
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--text);
      font-family: inherit;
      font-size: 12.5px;
      padding: 8px 10px;
      resize: vertical;
      min-height: 64px;
      outline: none;
      transition: border 0.15s;
    }
    textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-glow); }
    .row { display: flex; gap: 8px; align-items: center; }
    .col { flex: 1; display: flex; flex-direction: column; gap: 3px; }
    label { font-size: 11px; font-weight: 600; color: var(--text-muted); }
    select, input[type="text"], input[type="number"] {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--text);
      font-size: 12px;
      padding: 6px 8px;
      outline: none;
      font-family: inherit;
    }
    select:focus, input:focus { border-color: var(--accent); }
    input[type="range"] {
      accent-color: var(--accent);
      height: 4px;
      cursor: pointer;
    }
    .accordion {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--bg-card);
      overflow: hidden;
    }
    .accordion-header {
      padding: 8px 10px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      user-select: none;
      background: rgba(255,255,255,0.02);
    }
    .accordion-body {
      padding: 10px;
      border-top: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .lora-slot {
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 8px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .lora-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 11px;
    }
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: var(--radius);
      padding: 16px;
      text-align: center;
      background: rgba(255,255,255,0.01);
      cursor: pointer;
      transition: all 0.2s;
    }
    .dropzone.dragover { border-color: var(--accent); background: var(--accent-glow); }
    .btn-main {
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #fff;
      border: none;
      border-radius: var(--radius);
      font-weight: 700;
      font-size: 13px;
      padding: 10px 16px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      box-shadow: 0 4px 12px var(--accent-glow);
      transition: opacity 0.15s, transform 0.1s;
    }
    .btn-main:hover { opacity: 0.95; transform: translateY(-1px); }
    .btn-main:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .btn-sub {
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 11px;
      padding: 5px 9px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 500;
    }
    .btn-sub:hover { background: var(--bg-hover); border-color: var(--accent); }
    .btn-tag {
      background: rgba(6, 182, 212, 0.15);
      border: 1px solid rgba(6, 182, 212, 0.35);
      color: var(--accent-cyan);
      font-size: 11px;
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
    }
    .btn-tag:hover { background: rgba(6, 182, 212, 0.25); }
    .tag-chip {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 2px 6px;
      font-size: 11px;
      cursor: pointer;
      user-select: none;
    }
    .tag-chip.character { border-color: var(--accent-2); color: #f472b6; }
    .tag-chip.artist { border-color: var(--accent); color: #c084fc; }
    .tag-chip:hover { background: var(--accent-glow); }
    .viewer-pane {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: #05070a;
      overflow: hidden;
      position: relative;
    }
    .canvas-container {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      overflow: hidden;
      padding: 16px;
    }
    .canvas-container img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border-radius: 6px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.7);
    }
    .canvas-empty {
      color: var(--text-muted);
      font-size: 13px;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
    }
    .viewer-bar {
      height: 48px;
      background: var(--bg-panel);
      border-top: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      font-size: 12px;
    }
    .gallery-strip {
      height: 64px;
      background: var(--bg-card);
      border-top: 1px solid var(--border);
      display: flex;
      gap: 8px;
      padding: 6px 12px;
      overflow-x: auto;
      align-items: center;
    }
    .gallery-item {
      height: 52px;
      width: 52px;
      border-radius: 4px;
      overflow: hidden;
      cursor: pointer;
      border: 2px solid transparent;
      flex-shrink: 0;
    }
    .gallery-item.active { border-color: var(--accent); }
    .gallery-item img { width: 100%; height: 100%; object-fit: cover; }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      color: var(--text-muted);
    }
    .spinner {
      width: 12px;
      height: 12px;
      border: 2px solid rgba(255,255,255,0.2);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .tag-autocomplete-menu {
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      z-index: 9999;
      background: #11141a;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: 0 10px 30px rgba(0,0,0,0.8);
      max-height: 220px;
      overflow-y: auto;
      display: none;
    }
    .tag-ac-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 10px;
      font-size: 11.5px;
      cursor: pointer;
      border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .tag-ac-item:hover, .tag-ac-item.active {
      background: var(--accent-glow);
      color: #fff;
    }
    .tag-ac-badge {
      font-size: 9.5px;
      padding: 1px 5px;
      border-radius: 3px;
      font-weight: 600;
    }
    .tag-ac-badge.general { background: rgba(255,255,255,0.08); color: var(--text-muted); }
    .tag-ac-badge.character { background: rgba(236,72,153,0.25); color: #f472b6; border: 1px solid rgba(236,72,153,0.4); }
    .tag-ac-badge.artist { background: rgba(168,85,247,0.25); color: #c084fc; border: 1px solid rgba(168,85,247,0.4); }
    .tag-ac-badge.copyright { background: rgba(6,182,212,0.25); color: #22d3ee; border: 1px solid rgba(6,182,212,0.4); }
    .tag-ac-count { font-size: 10px; color: var(--text-muted); margin-left: 6px; }
    .toast-container {
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 999999;
      display: flex;
      flex-direction: column;
      gap: 10px;
      pointer-events: none;
    }
    .toast {
      background: #161b22;
      color: #f0f6fc;
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent);
      padding: 10px 16px;
      border-radius: var(--radius);
      font-size: 12.5px;
      font-weight: 500;
      box-shadow: 0 10px 30px rgba(0,0,0,0.7);
      opacity: 0;
      transform: translateY(12px) scale(0.98);
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      align-items: center;
      gap: 10px;
      max-width: 420px;
      pointer-events: auto;
    }
    .toast.show {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
    .toast.success { border-left-color: var(--success); }
    .toast.error { border-left-color: var(--danger); }
    .toast.info { border-left-color: var(--accent); }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span>✨ ANIMA WebUI</span>
      <span class="brand-badge">Qwen-3 DiT</span>
    </div>
    <div class="header-controls">
      <div class="preset-pill">
        <button type="button" id="btn-mode-turbo" onclick="setPreset('turbo')">⚡ Turbo (8 Steps)</button>
        <button type="button" id="btn-mode-base" class="active" onclick="setPreset('base')">🎨 Base (35 Steps)</button>
      </div>
      <button class="btn-sub" onclick="clearCache()" title="Clear GPU memory">🧹 VRAM</button>
    </div>
  </header>

  <div class="app-body">
    <!-- Controls (40%) -->
    <div class="controls-pane">
      <div class="tabs-bar">
        <button class="tab-btn active" onclick="switchTab('gen')">🎨 Txt2Img</button>
        <button class="tab-btn" onclick="switchTab('img2img')">🖼️ Img2Img</button>
        <button class="tab-btn" onclick="switchTab('tagger')">🏷️ Auto-Tagger</button>
        <button class="tab-btn" onclick="switchTab('down')">📥 Downloader</button>
      </div>

      <div class="controls-scroll">
        <!-- Common Prompt Section with Booru Autocomplete -->
        <div class="prompt-box" style="position:relative;">
          <div class="box-header">
            <span>Positive Prompt (Danbooru Autocomplete Active)</span>
            <button class="btn-sub" style="padding:1px 5px;" onclick="$('#prompt').value=''">✕</button>
          </div>
          <textarea id="prompt" rows="3" placeholder="Type anime tags (e.g. 1girl, @artist, hatsune miku)..." autocomplete="off" spellcheck="false"></textarea>
          <div id="tag-autocomplete" class="tag-autocomplete-menu"></div>
        </div>

        <!-- Top Action Row (Generate + Cancel) -->
        <div class="row" style="margin:2px 0 4px 0; gap:8px;">
          <button class="btn-main btn-generate-action" onclick="startGenerate()" style="flex:1; font-size:13.5px; padding:11px; font-weight:700;">
            <span>✨ Generate Image</span>
          </button>
          <button class="btn-cancel-action" onclick="cancelGenerate()" style="display:none; padding:11px 16px; background:#dc2626; color:#fff; border:none; border-radius:var(--radius); font-size:12.5px; font-weight:700; cursor:pointer;" title="Cancel running generation">
            <span>🛑 Cancel</span>
          </button>
        </div>

        <!-- Real-time Progress Bar -->
        <div id="gen-progress-wrap" style="display:none; margin:2px 0 6px 0; background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:8px 10px;">
          <div style="display:flex; justify-content:space-between; font-size:11px; font-weight:600; margin-bottom:4px;">
            <span id="gen-progress-step" style="color:var(--text);">Sampling...</span>
            <span id="gen-progress-pct" style="color:var(--accent); font-weight:700;">0%</span>
          </div>
          <div style="width:100%; height:7px; background:rgba(255,255,255,0.06); border-radius:4px; overflow:hidden;">
            <div id="gen-progress-fill" style="width:0%; height:100%; background:linear-gradient(90deg, var(--accent), var(--accent-2)); transition:width 0.25s ease; border-radius:4px;"></div>
          </div>
        </div>

        <div class="accordion">
          <div class="accordion-header" onclick="toggleAcc(this)">
            <span>Negative Prompt</span>
            <span>▾</span>
          </div>
          <div class="accordion-body" style="display:none;">
            <textarea id="negative" rows="2">worst quality, low quality, bad anatomy, bad hands, missing fingers, extra digits, signature, watermark</textarea>
          </div>
        </div>

        <!-- Tab 1: Txt2Img -->
        <div id="tab-gen" class="tab-content">
          <div class="row" style="margin-bottom:8px;">
            <div class="col">
              <label>Aspect Ratio</label>
              <select id="aspect_ratio" onchange="updateAspect()">
                <option value="1:1">1:1 Square (1024×1024)</option>
                <option value="3:4">3:4 Portrait (896×1152)</option>
                <option value="2:3">2:3 Anime Poster (832×1216)</option>
                <option value="9:16">9:16 Tall Mobile (768×1344)</option>
                <option value="4:3">4:3 Landscape (1152×896)</option>
                <option value="3:2">3:2 Anime Wide (1216×832)</option>
                <option value="16:9">16:9 Cinema (1344×768)</option>
              </select>
            </div>
            <div class="col">
              <label>Model Checkpoint</label>
              <select id="checkpoint"></select>
            </div>
          </div>

          <div class="row" style="margin-bottom:8px;">
            <div class="col">
              <label>Steps: <span id="val-steps">35</span></label>
              <input type="range" id="steps" min="4" max="60" value="35" oninput="$('#val-steps').innerText=this.value">
            </div>
            <div class="col">
              <label>CFG: <span id="val-cfg">4.5</span></label>
              <input type="range" id="cfg" min="1.0" max="10.0" step="0.5" value="4.5" oninput="$('#val-cfg').innerText=this.value">
            </div>
          </div>

          <div class="row" style="margin-bottom:8px;">
            <div class="col">
              <label>Sampler</label>
              <select id="sampler">
                <option value="er_sde" selected>er_sde</option>
                <option value="euler">euler</option>
                <option value="euler_ancestral">euler_ancestral</option>
                <option value="dpmpp_2m">dpmpp_2m</option>
                <option value="dpmpp_sde">dpmpp_sde</option>
                <option value="dpmpp_2m_sde">dpmpp_2m_sde</option>
              </select>
            </div>
            <div class="col">
              <label>Scheduler</label>
              <select id="scheduler">
                <option value="normal" selected>normal</option>
                <option value="simple">simple</option>
                <option value="karras">karras</option>
                <option value="sgm_uniform">sgm_uniform</option>
              </select>
            </div>
          </div>

          <div class="row" style="margin-bottom:8px;">
            <div class="col">
              <label>Seed (-1 = Random)</label>
              <input type="number" id="seed" value="-1">
            </div>
            <div class="col">
              <label>Batch Count</label>
              <input type="number" id="batch" min="1" max="8" value="1">
            </div>
          </div>
        </div>

        <!-- Tab 2: Img2Img -->
        <div id="tab-img2img" class="tab-content" style="display:none;">
          <div class="dropzone" id="i2i-dropzone" onclick="$('#i2i-file').click()">
            <input type="file" id="i2i-file" accept="image/*" style="display:none" onchange="handleFileSelect(this.files[0])">
            <div id="i2i-placeholder">📁 Drop reference image or click to browse</div>
            <img id="i2i-preview" style="max-height:140px; display:none; margin:0 auto; border-radius:4px;">
          </div>
          <div class="row" style="margin-top:8px;">
            <button class="btn-tag" onclick="tagCurrentImage()" style="flex:1;">🏷️ Auto-Tag Reference</button>
            <button class="btn-sub" onclick="clearI2I()">✕ Clear</button>
          </div>
          <div class="row" style="margin-top:8px;">
            <div class="col">
              <label>Denoise Strength: <span id="val-denoise">0.65</span></label>
              <input type="range" id="denoise" min="0.10" max="1.0" step="0.05" value="0.65" oninput="$('#val-denoise').innerText=this.value">
            </div>
          </div>
          <div class="row" style="margin-top:8px;">
            <div class="col">
              <label for="img2img_resize">Resize & Aspect Ratio</label>
              <select id="img2img_resize">
                <option value="crop" selected>📐 Crop & Fit to Selected Aspect Ratio</option>
                <option value="original">🖼️ Keep Source Aspect Ratio (Safe 1024px Max)</option>
                <option value="stretch">↔️ Stretch to Selected Aspect Ratio</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Tab 3: WD-EVA02 Auto-Tagger -->
        <div id="tab-tagger" class="tab-content" style="display:none;">
          <div class="dropzone" id="tagger-dropzone" onclick="$('#tagger-file').click()">
            <input type="file" id="tagger-file" accept="image/*" style="display:none" onchange="handleTaggerUpload(this.files[0])">
            <div id="tagger-placeholder">🏷️ Drop anime image to extract Danbooru tags</div>
            <img id="tagger-preview" style="max-height:140px; display:none; margin:0 auto; border-radius:4px;">
          </div>
          <div class="row" style="margin-top:8px;">
            <div class="col">
              <label>General Threshold: <span id="val-gth">0.35</span></label>
              <input type="range" id="tagger-gth" min="0.10" max="0.90" step="0.05" value="0.35" oninput="$('#val-gth').innerText=this.value">
            </div>
            <div class="col">
              <label>Character Threshold: <span id="val-cth">0.75</span></label>
              <input type="range" id="tagger-cth" min="0.20" max="0.95" step="0.05" value="0.75" oninput="$('#val-cth').innerText=this.value">
            </div>
          </div>
          <button class="btn-main" id="btn-run-tagger" onclick="runStandaloneTagger()" style="margin-top:8px; width:100%;">⚡ Extract Tags with WD-EVA02</button>
          <div id="tagger-results" style="margin-top:10px; display:none;">
            <div class="box-header" style="margin-bottom:6px;">
              <span>Extracted Tags:</span>
              <div class="row">
                <button class="btn-sub" onclick="copyExtractedTags()">📋 Copy</button>
                <button class="btn-sub" onclick="applyTagsToPrompt(false)">✨ Replace</button>
                <button class="btn-sub" onclick="applyTagsToPrompt(true)">➕ Append</button>
              </div>
            </div>
            <div id="tag-chips-container" style="display:flex; flex-wrap:wrap; gap:4px; max-height:160px; overflow-y:auto;"></div>
          </div>
        </div>

        <!-- Tab 4: Downloader -->
        <div id="tab-down" class="tab-content" style="display:none;">
          <div class="col" style="gap:8px;">
            <div style="font-weight:700; font-size:12.5px; color:var(--text);">📥 Download Models & LoRAs</div>
            <div style="font-size:11px; color:var(--text-muted);">Download diffusion models or LoRAs directly from Civitai, Hugging Face, or Google Drive into ComfyUI.</div>

            <label>Model / LoRA URL or Civitai AIR</label>
            <input type="text" id="down-url" placeholder="https://civitai.com/... or urn:air:... or huggingface.co/... or GDrive">

            <div class="row">
              <div class="col">
                <label>Save as</label>
                <select id="down-type">
                  <option value="auto">Auto-Detect</option>
                  <option value="lora">LoRA (models/loras)</option>
                  <option value="checkpoint">Diffusion Model (models/diffusion_models)</option>
                </select>
              </div>
              <div class="col">
                <label>Civitai Key (optional)</label>
                <input type="password" id="down-key" placeholder="API key if restricted">
              </div>
            </div>

            <div class="col">
              <label>Custom Filename (optional)</label>
              <input type="text" id="down-name" placeholder="custom_name.safetensors">
            </div>

            <button class="btn-main" id="btn-start-down" onclick="startDownload()" style="margin-top:4px;">⬇️ Download to WebUI</button>
            <div id="down-status" style="font-size:11px; color:var(--text-muted); margin-top:4px; word-break:break-all; min-height:20px;"></div>

            <div style="margin-top:6px; background:rgba(255,255,255,0.02); border:1px solid var(--border); border-radius:6px; padding:8px; font-size:11px; color:var(--text-muted); line-height:1.5;">
              <b style="color:var(--text);">💡 Supported formats:</b>
              <ul style="margin:4px 0 0; padding-left:16px;">
                <li><b>Civitai:</b> Model page URL (e.g. <code>civitai.com/models/...</code>) or AIR <code>urn:air:...</code></li>
                <li><b>Hugging Face:</b> Direct links (auto-converts <code>/blob/</code> to <code>/resolve/</code>)</li>
                <li><b>Google Drive:</b> Public shared file or folder links</li>
              </ul>
            </div>
          </div>
        </div>

        <!-- 6-Slot LoRA Rack -->
        <div class="accordion" style="margin-top:4px;">
          <div class="accordion-header" onclick="toggleAcc(this)">
            <span>🎛️ LoRA Rack (6 Slots Active)</span>
            <span id="lora-active-badge" style="font-size:10px; color:var(--accent);">0 active</span>
          </div>
          <div class="accordion-body" id="lora-rack-body" style="display:flex; flex-direction:column; gap:8px; padding:8px;">
            <div class="lora-slot" id="slot-card-1">
              <div class="lora-header">
                <label><input type="checkbox" id="lora-1-on" checked onchange="updateLoraBadge()"> Slot 1</label>
                <div class="row" style="gap:4px; align-items:center;">
                  <span style="font-size:10px; color:var(--text-muted);">Weight:</span>
                  <input type="number" id="lora-1-num" step="0.05" min="-5.0" max="10.0" value="1.0"
                    style="width:68px; padding:2px 4px; font-size:11px; font-weight:600; text-align:center; background:var(--bg-card); border:1px solid var(--border); border-radius:4px; color:var(--text);"
                    oninput="syncLoraWeight(1, 'num')">
                  <button type="button" class="btn-sub" style="padding:1px 5px; font-size:10px;" onclick="resetLoraSlot(1)" title="Reset to 1.0">↺</button>
                </div>
              </div>
              <div class="row" style="margin-top:2px;">
                <select id="lora-1-name" style="flex:1;" onchange="updateLoraBadge()">
                  <option value="None">None</option>
                </select>
                <input type="range" id="lora-1-str" min="-2.0" max="5.0" step="0.05" value="1.0" style="width:130px;"
                  oninput="syncLoraWeight(1, 'range')">
              </div>
            </div>
            <div class="lora-slot" id="slot-card-2">
              <div class="lora-header">
                <label><input type="checkbox" id="lora-2-on" checked onchange="updateLoraBadge()"> Slot 2</label>
                <div class="row" style="gap:4px; align-items:center;">
                  <span style="font-size:10px; color:var(--text-muted);">Weight:</span>
                  <input type="number" id="lora-2-num" step="0.05" min="-5.0" max="10.0" value="1.0"
                    style="width:68px; padding:2px 4px; font-size:11px; font-weight:600; text-align:center; background:var(--bg-card); border:1px solid var(--border); border-radius:4px; color:var(--text);"
                    oninput="syncLoraWeight(2, 'num')">
                  <button type="button" class="btn-sub" style="padding:1px 5px; font-size:10px;" onclick="resetLoraSlot(2)" title="Reset to 1.0">↺</button>
                </div>
              </div>
              <div class="row" style="margin-top:2px;">
                <select id="lora-2-name" style="flex:1;" onchange="updateLoraBadge()">
                  <option value="None">None</option>
                </select>
                <input type="range" id="lora-2-str" min="-2.0" max="5.0" step="0.05" value="1.0" style="width:130px;"
                  oninput="syncLoraWeight(2, 'range')">
              </div>
            </div>
            <div class="lora-slot" id="slot-card-3">
              <div class="lora-header">
                <label><input type="checkbox" id="lora-3-on" checked onchange="updateLoraBadge()"> Slot 3</label>
                <div class="row" style="gap:4px; align-items:center;">
                  <span style="font-size:10px; color:var(--text-muted);">Weight:</span>
                  <input type="number" id="lora-3-num" step="0.05" min="-5.0" max="10.0" value="1.0"
                    style="width:68px; padding:2px 4px; font-size:11px; font-weight:600; text-align:center; background:var(--bg-card); border:1px solid var(--border); border-radius:4px; color:var(--text);"
                    oninput="syncLoraWeight(3, 'num')">
                  <button type="button" class="btn-sub" style="padding:1px 5px; font-size:10px;" onclick="resetLoraSlot(3)" title="Reset to 1.0">↺</button>
                </div>
              </div>
              <div class="row" style="margin-top:2px;">
                <select id="lora-3-name" style="flex:1;" onchange="updateLoraBadge()">
                  <option value="None">None</option>
                </select>
                <input type="range" id="lora-3-str" min="-2.0" max="5.0" step="0.05" value="1.0" style="width:130px;"
                  oninput="syncLoraWeight(3, 'range')">
              </div>
            </div>
            <div class="lora-slot" id="slot-card-4">
              <div class="lora-header">
                <label><input type="checkbox" id="lora-4-on" checked onchange="updateLoraBadge()"> Slot 4</label>
                <div class="row" style="gap:4px; align-items:center;">
                  <span style="font-size:10px; color:var(--text-muted);">Weight:</span>
                  <input type="number" id="lora-4-num" step="0.05" min="-5.0" max="10.0" value="1.0"
                    style="width:68px; padding:2px 4px; font-size:11px; font-weight:600; text-align:center; background:var(--bg-card); border:1px solid var(--border); border-radius:4px; color:var(--text);"
                    oninput="syncLoraWeight(4, 'num')">
                  <button type="button" class="btn-sub" style="padding:1px 5px; font-size:10px;" onclick="resetLoraSlot(4)" title="Reset to 1.0">↺</button>
                </div>
              </div>
              <div class="row" style="margin-top:2px;">
                <select id="lora-4-name" style="flex:1;" onchange="updateLoraBadge()">
                  <option value="None">None</option>
                </select>
                <input type="range" id="lora-4-str" min="-2.0" max="5.0" step="0.05" value="1.0" style="width:130px;"
                  oninput="syncLoraWeight(4, 'range')">
              </div>
            </div>
            <div class="lora-slot" id="slot-card-5">
              <div class="lora-header">
                <label><input type="checkbox" id="lora-5-on" checked onchange="updateLoraBadge()"> Slot 5</label>
                <div class="row" style="gap:4px; align-items:center;">
                  <span style="font-size:10px; color:var(--text-muted);">Weight:</span>
                  <input type="number" id="lora-5-num" step="0.05" min="-5.0" max="10.0" value="1.0"
                    style="width:68px; padding:2px 4px; font-size:11px; font-weight:600; text-align:center; background:var(--bg-card); border:1px solid var(--border); border-radius:4px; color:var(--text);"
                    oninput="syncLoraWeight(5, 'num')">
                  <button type="button" class="btn-sub" style="padding:1px 5px; font-size:10px;" onclick="resetLoraSlot(5)" title="Reset to 1.0">↺</button>
                </div>
              </div>
              <div class="row" style="margin-top:2px;">
                <select id="lora-5-name" style="flex:1;" onchange="updateLoraBadge()">
                  <option value="None">None</option>
                </select>
                <input type="range" id="lora-5-str" min="-2.0" max="5.0" step="0.05" value="1.0" style="width:130px;"
                  oninput="syncLoraWeight(5, 'range')">
              </div>
            </div>
            <div class="lora-slot" id="slot-card-6">
              <div class="lora-header">
                <label><input type="checkbox" id="lora-6-on" checked onchange="updateLoraBadge()"> Slot 6</label>
                <div class="row" style="gap:4px; align-items:center;">
                  <span style="font-size:10px; color:var(--text-muted);">Weight:</span>
                  <input type="number" id="lora-6-num" step="0.05" min="-5.0" max="10.0" value="1.0"
                    style="width:68px; padding:2px 4px; font-size:11px; font-weight:600; text-align:center; background:var(--bg-card); border:1px solid var(--border); border-radius:4px; color:var(--text);"
                    oninput="syncLoraWeight(6, 'num')">
                  <button type="button" class="btn-sub" style="padding:1px 5px; font-size:10px;" onclick="resetLoraSlot(6)" title="Reset to 1.0">↺</button>
                </div>
              </div>
              <div class="row" style="margin-top:2px;">
                <select id="lora-6-name" style="flex:1;" onchange="updateLoraBadge()">
                  <option value="None">None</option>
                </select>
                <input type="range" id="lora-6-str" min="-2.0" max="5.0" step="0.05" value="1.0" style="width:130px;"
                  oninput="syncLoraWeight(6, 'range')">
              </div>
            </div>
          </div>
        </div>

                <!-- Bottom Action Row (Generate + Cancel) -->
        <div class="row" style="margin-top:8px; gap:8px;">
          <button class="btn-main btn-generate-action" id="btn-generate" onclick="startGenerate()" style="flex:1; font-size:13.5px; padding:11px; font-weight:700;">
            <span>✨ Generate Image</span>
          </button>
          <button class="btn-cancel-action" onclick="cancelGenerate()" style="display:none; padding:11px 16px; background:#dc2626; color:#fff; border:none; border-radius:var(--radius); font-size:12.5px; font-weight:700; cursor:pointer;" title="Cancel running generation">
            <span>🛑 Cancel</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Viewer (60%) -->
    <div class="viewer-pane">
      <div class="canvas-container">
        <div id="canvas-empty" class="canvas-empty">
          <div style="font-size:36px;">🌸</div>
          <div>Ready to create with Anima</div>
          <div style="font-size:11px; opacity:0.6;">Press Ctrl+Enter or click Generate</div>
        </div>
        <img id="main-canvas" style="display:none;" alt="Generated Artwork">
      </div>

      <div class="viewer-bar">
        <div class="status-pill" id="gen-status">
          <span id="status-text">Idle</span>
        </div>
        <div class="row">
          <button class="btn-sub" id="btn-dl" style="display:none;" onclick="downloadCurrentImage()">💾 Save PNG</button>
          <button class="btn-sub" id="btn-copy-seed" style="display:none;" onclick="copySeed()">🌱 Seed</button>
          <button class="btn-sub" id="btn-to-i2i" style="display:none;" onclick="sendToImg2Img()">➡️ Send to Img2Img</button>
        </div>
      </div>

      <div class="gallery-strip" id="gallery-strip"></div>
    </div>
  </div>

  <script>
    const $ = s => document.querySelector(s);
    const $$ = s => document.querySelectorAll(s);

    let gState = {
      models: [], loras: [], activeTab: 'gen',
      inputImage: null, taggerImage: null,
      currentPromptId: null, currentSeed: null,
      currentFilename: null, history: []
    };

    function toggleAcc(el) {
      const body = el.nextElementSibling;
      body.display = (body.style.display === 'none') ? 'flex' : 'none';
      body.style.display = body.display;
    }

    function switchTab(t) {
      gState.activeTab = t;
      $$('.tab-btn').forEach(b => b.classList.remove('active'));
      $$('.tab-content').forEach(c => c.style.display = 'none');
      if (t === 'gen') {
        $$('.tab-btn')[0].classList.add('active');
        $('#tab-gen').style.display = 'block';
      } else if (t === 'img2img') {
        $$('.tab-btn')[1].classList.add('active');
        $('#tab-img2img').style.display = 'block';
      } else if (t === 'tagger') {
        $$('.tab-btn')[2].classList.add('active');
        $('#tab-tagger').style.display = 'block';
      } else if (t === 'down') {
        $$('.tab-btn')[3].classList.add('active');
        $('#tab-down').style.display = 'block';
      }
    }

    function setPreset(mode) {
      if (mode === 'turbo') {
        $('#btn-mode-turbo').classList.add('active');
        $('#btn-mode-base').classList.remove('active');
        $('#steps').value = 8;
        $('#val-steps').innerText = 8;
        $('#cfg').value = 1.0;
        $('#val-cfg').innerText = 1.0;
        // Auto select turbo lora if available
        const turboLora = gState.loras.find(l => l.toLowerCase().includes('turbo'));
        if (turboLora) {
          $('#lora-1-name').value = turboLora;
          $('#lora-1-on').checked = true;
          if ($('#lora-1-num')) $('#lora-1-num').value = 1.0;
          if ($('#lora-1-str')) $('#lora-1-str').value = 1.0;
          updateLoraBadge();
        }
      } else {
        $('#btn-mode-base').classList.add('active');
        $('#btn-mode-turbo').classList.remove('active');
        $('#steps').value = 35;
        $('#val-steps').innerText = 35;
        $('#cfg').value = 4.5;
        $('#val-cfg').innerText = 4.5;
      }
    }

    async function loadModels() {
      try {
        const res = await fetch('/api/models');
        const data = await res.json();
        gState.models = data.models || [];
        gState.loras = data.loras || [];

        // Checkpoint dropdown
        const cp = $('#checkpoint');
        const curCp = cp.value;
        cp.innerHTML = gState.models.map(m => `<option value="${m}">${m}</option>`).join('');
        if (gState.models.includes(curCp)) cp.value = curCp;

        // Sampler & Scheduler (er_sde preferred for Anima)
        const sm = $('#sampler');
        let samplers = data.samplers || [];
        if (samplers.includes('er_sde')) {
          samplers = ['er_sde', ...samplers.filter(s => s !== 'er_sde')];
        }
        const curSm = sm.value;
        sm.innerHTML = samplers.map(s => `<option value="${s}">${s}</option>`).join('');
        if (curSm && curSm !== 'euler' && samplers.includes(curSm)) {
          sm.value = curSm;
        } else if (samplers.includes('er_sde')) {
          sm.value = 'er_sde';
        }

        $('#scheduler').innerHTML = data.schedulers.map(s => `<option value="${s}">${s}</option>`).join('');

        // Populate all 6 LoRA select dropdowns
        populateLoraSlots();
      } catch (e) {
        console.error("Failed loading models", e);
      }
    }

    function populateLoraSlots() {
      for (let i = 1; i <= 6; i++) {
        const sel = $(`#lora-${i}-name`);
        if (sel) {
          const curL = sel.value;
          sel.innerHTML = gState.loras.map(l => `<option value="${l}">${l}</option>`).join('');
          if (curL && gState.loras.includes(curL)) {
            sel.value = curL;
          } else if (i === 1) {
            const turbo = gState.loras.find(l => l.toLowerCase().includes('turbo'));
            if (turbo) sel.value = turbo;
          }
        }
      }
      updateLoraBadge();
    }

    function syncLoraWeight(i, source) {
      const numEl = $(`#lora-${i}-num`);
      const rangeEl = $(`#lora-${i}-str`);
      if (source === 'range') {
        numEl.value = rangeEl.value;
      } else {
        const val = parseFloat(numEl.value);
        if (!isNaN(val)) {
          rangeEl.value = Math.max(-2.0, Math.min(5.0, val));
        }
      }
    }

    function resetLoraSlot(i) {
      const numEl = $(`#lora-${i}-num`);
      const rangeEl = $(`#lora-${i}-str`);
      numEl.value = 1.0;
      rangeEl.value = 1.0;
    }

    function updateLoraBadge() {
      let count = 0;
      for (let i = 1; i <= 6; i++) {
        const sel = $(`#lora-${i}-name`);
        const chk = $(`#lora-${i}-on`);
        if (sel && sel.value !== 'None' && chk && chk.checked) count++;
      }
      $('#lora-active-badge').innerText = `${count} active`;
    }

    // Reference Image Upload
    async function handleFileSelect(file) {
      if (!file) return;
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/upload', { method: 'POST', body: form });
      const data = await res.json();
      if (data.filename) {
        gState.inputImage = data.filename;
        $('#i2i-placeholder').style.display = 'none';
        $('#i2i-preview').src = `/api/view?filename=${data.filename}&type=input`;
        $('#i2i-preview').style.display = 'block';
      }
    }

    function clearI2I() {
      gState.inputImage = null;
      $('#i2i-preview').style.display = 'none';
      $('#i2i-placeholder').style.display = 'block';
      $('#i2i-file').value = '';
    }

    // Auto-Tagger handler
    async function handleTaggerUpload(file) {
      if (!file) return;
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/upload', { method: 'POST', body: form });
      const data = await res.json();
      if (data.filename) {
        gState.taggerImage = data.filename;
        $('#tagger-placeholder').style.display = 'none';
        $('#tagger-preview').src = `/api/view?filename=${data.filename}&type=input`;
        $('#tagger-preview').style.display = 'block';
      }
    }

    async function tagCurrentImage() {
      if (!gState.inputImage) return showToast("Please upload a reference image first.", "error");
      runTagger(gState.inputImage, true);
    }

    async function runStandaloneTagger() {
      if (!gState.taggerImage) return showToast("Please upload an image to tag.", "error");
      runTagger(gState.taggerImage, false);
    }

    async function runTagger(fn, autoInsert) {
      const btn = $('#btn-run-tagger');
      btn.disabled = true;
      btn.innerText = '⏳ Analyzing with WD-EVA02...';
      try {
        const res = await fetch('/api/tag_image', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            filename: fn,
            general_threshold: parseFloat($('#tagger-gth').value),
            character_threshold: parseFloat($('#tagger-cth').value)
          })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        window._lastExtractedCaption = data.caption;
        if (autoInsert) {
          const p = $('#prompt');
          p.value = p.value ? `${p.value}, ${data.caption}` : data.caption;
          showToast("Extracted Danbooru tags added to Prompt!", "success");
        } else {
          $('#tagger-results').style.display = 'block';
          const container = $('#tag-chips-container');
          let chips = '';
          (data.characters || []).forEach(c => { chips += `<span class="tag-chip character" onclick="toggleTag('${c[0]}')">👤 ${c[0]} (${c[1]})</span>`; });
          (data.artists || []).forEach(a => { chips += `<span class="tag-chip artist" onclick="toggleTag('${a[0]}')">🎨 ${a[0]} (${a[1]})</span>`; });
          (data.general || []).forEach(g => { chips += `<span class="tag-chip" onclick="toggleTag('${g[0]}')">${g[0]} (${g[1]})</span>`; });
          container.innerHTML = chips;
        }
      } catch (e) {
        showToast("Tagging error: " + e.message, "error", 4500);
      } finally {
        btn.disabled = false;
        btn.innerText = '⚡ Extract Tags with WD-EVA02';
      }
    }

    function toggleTag(t) {
      const p = $('#prompt');
      const val = p.value.trim();
      if (val.includes(t)) {
        p.value = val.replace(t, '').replace(/,\s*,/g, ',').replace(/^,\s*|,\s*$/g, '');
      } else {
        p.value = val ? `${val}, ${t}` : t;
      }
    }

    function copyExtractedTags() {
      if (window._lastExtractedCaption) {
        navigator.clipboard.writeText(window._lastExtractedCaption);
        showToast("Copied tags to clipboard!", "success");
      }
    }

    function applyTagsToPrompt(append) {
      if (!window._lastExtractedCaption) return;
      const p = $('#prompt');
      p.value = append && p.value ? `${p.value}, ${window._lastExtractedCaption}` : window._lastExtractedCaption;
    }

    // Generate Pipeline with Real-time Progress & Cancellation
    let pollTimer = null;
    let isGenerating = false;

    function setGeneratingState(generating, stepText, pct) {
      isGenerating = generating;
      $$('.btn-generate-action').forEach(b => {
        b.disabled = generating;
        if (generating) {
          const statusText = pct !== undefined ? `Generating (${pct}%)...` : 'Generating...';
          b.innerHTML = `<span class="spinner"></span> <span>${statusText}</span>`;
        } else {
          b.innerHTML = '<span>✨ Generate Image</span>';
        }
      });

      $$('.btn-cancel-action').forEach(b => {
        b.style.display = generating ? 'block' : 'none';
      });

      const progWrap = $('#gen-progress-wrap');
      if (progWrap) {
        progWrap.style.display = generating ? 'block' : 'none';
        if (pct !== undefined) {
          $('#gen-progress-fill').style.width = pct + '%';
          $('#gen-progress-pct').innerText = pct + '%';
        }
        if (stepText) {
          $('#gen-progress-step').innerText = stepText;
        }
      }

      $('#gen-status').innerHTML = generating ? '<span class="spinner"></span> <span>Generating...</span>' : '<span>Ready</span>';
    }

    async function cancelGenerate() {
      clearTimeout(pollTimer);
      showToast("Canceling generation...", "info", 1500);
      try {
        await fetch('/api/cancel', { method: 'POST' });
        showToast("Generation canceled.", "info");
      } catch (e) {
        console.error("Cancel error", e);
      }
      setGeneratingState(false);
      $('#gen-status').innerHTML = '<span>Canceled</span>';
    }

    async function startGenerate() {
      const prompt = $('#prompt').value.trim();
      if (!prompt) return showToast("Please enter a prompt.", "error");

      const payload = {
        prompt: prompt,
        negative: $('#negative').value.trim(),
        steps: parseInt($('#steps').value),
        cfg: parseFloat($('#cfg').value),
        sampler: $('#sampler').value,
        scheduler: $('#scheduler').value,
        aspect_ratio: $('#aspect_ratio').value,
        checkpoint: $('#checkpoint').value,
        seed: parseInt($('#seed').value),
        tab: gState.activeTab,
        input_image: gState.inputImage,
        denoise: parseFloat($('#denoise').value),
        img2img_resize: $('#img2img_resize') ? $('#img2img_resize').value : 'crop'
      };

      for (let i = 1; i <= 6; i++) {
        payload[`lora${i}`] = $(`#lora-${i}-name`) ? $(`#lora-${i}-name`).value : 'None';
        const numVal = $(`#lora-${i}-num`) ? parseFloat($(`#lora-${i}-num`).value) : 1.0;
        payload[`lora${i}_strength`] = isNaN(numVal) ? 1.0 : numVal;
        payload[`lora${i}_enabled`] = $(`#lora-${i}-on`) ? $(`#lora-${i}-on`).checked : false;
      }

      setGeneratingState(true, 'Initializing...', 0);

      try {
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        gState.currentPromptId = data.prompt_id;
        gState.currentSeed = data.seed;
        pollStatus(data.prompt_id);
      } catch (e) {
        showToast("Generation failed: " + e.message, "error", 6000);
        setGeneratingState(false);
      }
    }

    async function pollStatus(pid) {
      if (!isGenerating) return;
      try {
        const res = await fetch(`/api/status/${pid}`);
        const data = await res.json();
        if (data.state === 'done') {
          setGeneratingState(false);
          $('#gen-status').innerHTML = '<span>✓ Done</span>';
          const img = data.images[0];
          gState.currentFilename = img.filename;
          gState.currentSeed = data.seed;
          showImage(img.filename, data.seed);
          showToast("Image generation complete!", "success");
        } else if (data.state === 'error') {
          throw new Error(data.error || 'Generation failed');
        } else {
          if (data.progress && data.progress.max > 0) {
            const val = data.progress.value;
            const max = data.progress.max;
            const pct = data.progress.percent;
            setGeneratingState(true, `Step ${val}/${max}`, pct);
          } else if (data.position) {
            setGeneratingState(true, `Queue position: ${data.position}`, 0);
          }
          pollTimer = setTimeout(() => pollStatus(pid), 400);
        }
      } catch (e) {
        showToast("Status error: " + e.message, "error", 4500);
        setGeneratingState(false);
      }
    }

    function showImage(filename, seed) {
      const url = `/api/view?filename=${filename}`;
      $('#canvas-empty').style.display = 'none';
      const mc = $('#main-canvas');
      mc.src = url;
      mc.style.display = 'block';

      $('#btn-dl').style.display = 'inline-block';
      $('#btn-copy-seed').style.display = 'inline-block';
      $('#btn-to-i2i').style.display = 'inline-block';

      // Add to gallery
      const strip = $('#gallery-strip');
      const item = document.createElement('div');
      item.className = 'gallery-item active';
      item.innerHTML = `<img src="${url}">`;
      item.onclick = () => {
        $$('.gallery-item').forEach(g => g.classList.remove('active'));
        item.classList.add('active');
        mc.src = url;
        gState.currentFilename = filename;
        gState.currentSeed = seed;
      };
      $$('.gallery-item').forEach(g => g.classList.remove('active'));
      strip.prepend(item);
    }

    function downloadCurrentImage() {
      if (!gState.currentFilename) return;
      const a = document.createElement('a');
      a.href = `/api/view?filename=${gState.currentFilename}`;
      a.download = `Anima_${gState.currentFilename.replace('.png', '')}_seed${gState.currentSeed}.png`;
      a.click();
    }

    function copySeed() {
      if (gState.currentSeed !== null) {
        navigator.clipboard.writeText(String(gState.currentSeed));
        showToast(`Seed ${gState.currentSeed} copied!`, "success");
      }
    }

    async function sendToImg2Img() {
      if (!gState.currentFilename) return;
      try {
        const res = await fetch('/api/copy_to_input', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ filename: gState.currentFilename })
        });
        const data = await res.json();
        gState.inputImage = data.filename || gState.currentFilename;
      } catch (e) {
        gState.inputImage = gState.currentFilename;
      }
      switchTab('img2img');
      $('#i2i-placeholder').style.display = 'none';
      $('#i2i-preview').src = `/api/view?filename=${gState.inputImage}&type=input`;
      $('#i2i-preview').style.display = 'block';
    }

    async function clearCache() {
      await fetch('/api/clear_cache', { method: 'POST' });
      showToast("GPU VRAM cleared.", "info");
    }

    function playSuccessChime() {
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const now = ctx.currentTime;
        const osc1 = ctx.createOscillator();
        const osc2 = ctx.createOscillator();
        const gain = ctx.createGain();

        osc1.type = 'sine';
        osc1.frequency.setValueAtTime(587.33, now); // D5
        osc1.frequency.setValueAtTime(880, now + 0.12); // A5

        osc2.type = 'triangle';
        osc2.frequency.setValueAtTime(880, now + 0.12);
        osc2.frequency.setValueAtTime(1174.66, now + 0.24); // D6

        gain.gain.setValueAtTime(0.12, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.6);

        osc1.connect(gain);
        osc2.connect(gain);
        gain.connect(ctx.destination);

        osc1.start(now);
        osc2.start(now + 0.12);
        osc1.stop(now + 0.6);
        osc2.stop(now + 0.6);
      } catch (e) {}
    }

    // Downloader
    let dlPollTimer = null;
    async function startDownload() {
      const url = $('#down-url').value.trim();
      if (!url) return showToast("Please enter a URL or Civitai AIR identifier.", "error");
      const btn = $('#btn-start-down');
      const statusEl = $('#down-status');
      btn.disabled = true;
      statusEl.style.color = "var(--accent)";
      statusEl.innerText = "Initiating download...";

      try {
        const res = await fetch('/api/download', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            url: url,
            type: $('#down-type').value,
            api_key: $('#down-key').value.trim(),
            filename: $('#down-name').value.trim()
          })
        });
        const data = await res.json();
        if (data.error) {
          btn.disabled = false;
          statusEl.style.color = "var(--danger)";
          statusEl.innerText = "✗ " + data.error;
          showToast(data.error, "error", 4500);
        } else {
          pollDownload();
        }
      } catch (err) {
        btn.disabled = false;
        statusEl.style.color = "var(--danger)";
        statusEl.innerText = "Failed to start download: " + err;
      }
    }

    async function pollDownload() {
      const btn = $('#btn-start-down');
      const statusEl = $('#down-status');
      try {
        const res = await fetch('/api/download_status');
        const st = await res.json();
        if (!st) return;

        if (st.status === 'downloading') {
          btn.disabled = true;
          statusEl.style.color = "var(--accent)";
          statusEl.innerText = `⏳ [${st.percent || 0}%] ${st.msg || 'Downloading...'}`;
          dlPollTimer = setTimeout(pollDownload, 1000);
        } else if (st.status === 'done') {
          btn.disabled = false;
          statusEl.style.color = "var(--success)";
          statusEl.innerText = st.msg;
          playSuccessChime();

          // Refresh models and select newly downloaded item
          await loadModels();
          if (st.target_type === 'checkpoint' && st.filename) {
            $('#checkpoint').value = st.filename;
          } else if (st.target_type === 'lora' && st.filename) {
            $('#lora-1-name').value = st.filename;
            $('#lora-1-on').checked = true;
            updateLoraBadge();
          }
          showToast(st.msg, "error", 4500);
        } else if (st.status === 'error') {
          btn.disabled = false;
          statusEl.style.color = "var(--danger)";
          statusEl.innerText = "✗ " + (st.error || 'Download failed');
        } else {
          btn.disabled = false;
        }
      } catch (e) {
        btn.disabled = false;
        statusEl.innerText = "Download tracking error: " + e;
      }
    }

    // Hotkey: Ctrl+Enter to generate
    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        startGenerate();
      }
    });


    // ---------------------------------------------------------------------------

    // ---------------------------------------------------------------------------
    // A1111-style Prompt Weight Adjustment (Ctrl+Shift+Up / Ctrl+Shift+Down)
    // ---------------------------------------------------------------------------
    function adjustPromptWeight(textarea, delta) {
      if (!textarea) return;
      let start = textarea.selectionStart;
      let end = textarea.selectionEnd;
      const val = textarea.value;

      // 1. If no text selected, auto-expand to surrounding token or parenthesized group
      if (start === end) {
        let openParen = -1;
        let closeParen = -1;
        for (let i = start - 1; i >= 0; i--) {
          if (val[i] === ')') break;
          if (val[i] === '(') { openParen = i; break; }
        }
        if (openParen !== -1) {
          for (let i = start; i < val.length; i++) {
            if (val[i] === '(') break;
            if (val[i] === ')') { closeParen = i + 1; break; }
          }
        }

        if (openParen !== -1 && closeParen !== -1) {
          start = openParen;
          end = closeParen;
        } else {
          let tokenStart = start;
          while (tokenStart > 0 && val[tokenStart - 1] !== ',' && val[tokenStart - 1] !== '\n') {
            tokenStart--;
          }
          let tokenEnd = end;
          while (tokenEnd < val.length && val[tokenEnd] !== ',' && val[tokenEnd] !== '\n') {
            tokenEnd++;
          }
          const raw = val.slice(tokenStart, tokenEnd);
          const leadingSpaces = (raw.match(/^\s*/) || [''])[0].length;
          const trailingSpaces = (raw.match(/\s*$/) || [''])[0].length;
          start = tokenStart + leadingSpaces;
          end = tokenEnd - trailingSpaces;
        }
      }

      if (start >= end) return;

      const selectedText = val.slice(start, end).trim();
      if (!selectedText) return;

      let text = selectedText;
      let currentWeight = 1.0;

      // Match (tag:weight)
      const matchWeight = selectedText.match(/^\((.+):(-?\d*(?:\.\d+)?)\)$/);
      // Match (tag)
      const matchParen = selectedText.match(/^\((.+)\)$/);

      if (matchWeight) {
        text = matchWeight[1].trim();
        currentWeight = parseFloat(matchWeight[2]);
        if (isNaN(currentWeight)) currentWeight = 1.0;
      } else if (matchParen) {
        text = matchParen[1].trim();
        currentWeight = 1.1;
      }

      let newWeight = Math.round((currentWeight + delta) * 100) / 100;
      let replacement = '';
      if (Math.abs(newWeight - 1.0) < 0.001) {
        replacement = text;
      } else {
        const weightStr = (Math.round(newWeight * 100) / 100).toString();
        replacement = `(${text}:${weightStr})`;
      }

      const before = val.slice(0, start);
      const after = val.slice(end);
      textarea.value = before + replacement + after;

      // Keep replacement selected so user can press shortcut repeatedly
      textarea.selectionStart = start;
      textarea.selectionEnd = start + replacement.length;
      textarea.focus();
    }

    // Booru Tag Autocomplete Controller
    // ---------------------------------------------------------------------------
    const promptEl = $('#prompt');
    const acMenu = $('#tag-autocomplete');
    let acItems = [];
    let acActiveIdx = -1;
    let acDebounce = null;

    function getActiveTokenInfo() {
      const val = promptEl.value;
      const pos = promptEl.selectionStart;
      const beforeCursor = val.slice(0, pos);
      const lastDelim = Math.max(beforeCursor.lastIndexOf(','), beforeCursor.lastIndexOf('\n'));
      const tokenStart = lastDelim === -1 ? 0 : lastDelim + 1;
      const token = beforeCursor.slice(tokenStart).trim();
      return { token, tokenStart, pos };
    }

    if (promptEl) {
      promptEl.addEventListener('input', () => {
        clearTimeout(acDebounce);
        acDebounce = setTimeout(async () => {
          const { token } = getActiveTokenInfo();
          if (!token || token.length < 2) {
            hideAutocomplete();
            return;
          }
          try {
            const res = await fetch('/api/tags?q=' + encodeURIComponent(token));
            const data = await res.json();
            renderAutocomplete(data.tags || []);
          } catch (e) {
            hideAutocomplete();
          }
        }, 120);
      });

      promptEl.addEventListener('keydown', e => {
        // Ctrl+Up / Ctrl+Down (or Ctrl+Shift+Up / Ctrl+Shift+Down) to adjust tag weights (like A1111)
        if (e.ctrlKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
          e.preventDefault();
          adjustPromptWeight(promptEl, e.key === 'ArrowUp' ? 0.1 : -0.1);
          return;
        }
        if (e.ctrlKey && e.key === 'Enter') {
          e.preventDefault();
          startGenerate();
          return;
        }

        if (acMenu.style.display !== 'block') return;
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          acActiveIdx = (acActiveIdx + 1) % acItems.length;
          updateActiveAcItem();
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          acActiveIdx = (acActiveIdx - 1 + acItems.length) % acItems.length;
          updateActiveAcItem();
        } else if (e.key === 'Enter' || e.key === 'Tab') {
          if (acActiveIdx >= 0 && acActiveIdx < acItems.length) {
            e.preventDefault();
            selectAcTag(acItems[acActiveIdx].name);
          }
        } else if (e.key === 'Escape') {
          hideAutocomplete();
        }
      });
    }


      const negEl = $('#negative');
      if (negEl) {
        negEl.addEventListener('keydown', e => {
          if (e.ctrlKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
            e.preventDefault();
            adjustPromptWeight(negEl, e.key === 'ArrowUp' ? 0.1 : -0.1);
          } else if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            startGenerate();
          }
        });
      }

    function renderAutocomplete(tags) {
      if (!tags.length) {
        hideAutocomplete();
        return;
      }
      acItems = tags;
      acActiveIdx = 0;
      let html = '';
      tags.forEach((t, idx) => {
        const catClass = t.cat === 4 ? 'character' : (t.cat === 1 ? 'artist' : (t.cat === 3 ? 'copyright' : 'general'));
        const catLabel = t.cat === 4 ? 'char' : (t.cat === 1 ? 'artist' : (t.cat === 3 ? 'series' : 'tag'));
        const countFmt = t.count > 1000000 ? `${(t.count/1000000).toFixed(1)}M` : (t.count > 1000 ? `${(t.count/1000).toFixed(0)}k` : t.count);
        html += `
          <div class="tag-ac-item ${idx === 0 ? 'active' : ''}" onclick="selectAcTag('${t.name}')">
            <span><b>${t.name}</b> <span class="tag-ac-count">${countFmt}</span></span>
            <span class="tag-ac-badge ${catClass}">${catLabel}</span>
          </div>
        `;
      });
      acMenu.innerHTML = html;
      acMenu.style.display = 'block';
    }

    function updateActiveAcItem() {
      const items = acMenu.querySelectorAll('.tag-ac-item');
      items.forEach((item, idx) => {
        item.classList.toggle('active', idx === acActiveIdx);
        if (idx === acActiveIdx) item.scrollIntoView({ block: 'nearest' });
      });
    }

    function selectAcTag(tag) {
      const { tokenStart, pos } = getActiveTokenInfo();
      const val = promptEl.value;
      const before = val.slice(0, tokenStart);
      const after = val.slice(pos);
      const prefix = (before && !before.endsWith(' ') && !before.endsWith('\n')) ? ' ' : '';
      promptEl.value = before + prefix + tag + ', ' + after.replace(/^[\s,]+/, '');
      hideAutocomplete();
      promptEl.focus();
      const newCursor = (before + prefix + tag + ', ').length;
      promptEl.setSelectionRange(newCursor, newCursor);
    }

    function hideAutocomplete() {
      if (acMenu) {
        acMenu.style.display = 'none';
        acItems = [];
        acActiveIdx = -1;
      }
    }

    document.addEventListener('click', e => {
      if (acMenu && !acMenu.contains(e.target) && e.target !== promptEl) {
        hideAutocomplete();
      }
    });


    // ---------------------------------------------------------------------------
    // Non-blocking Toast Notification System (replaces native alert popups)
    // ---------------------------------------------------------------------------
    function showToast(msg, type = 'info', duration = 3000) {
      const container = $('#toast-container');
      if (!container) return;
      const toast = document.createElement('div');
      toast.className = `toast ${type}`;
      const icon = type === 'success' ? '✓' : (type === 'error' ? '✕' : 'ℹ');
      toast.innerHTML = `<span style="font-weight:700;">${icon}</span> <span>${msg}</span>`;
      container.appendChild(toast);
      setTimeout(() => toast.classList.add('show'), 10);
      setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 250);
      }, duration);
    }

    // ---------------------------------------------------------------------------
    // Drag-and-Drop & Clipboard Paste Controller
    // ---------------------------------------------------------------------------
    function initDragAndDrop() {
      // 1. Prevent default browser drag/drop navigation everywhere (stops new tab open)
      ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
        window.addEventListener(name, e => {
          e.preventDefault();
          e.stopPropagation();
        }, false);
        document.addEventListener(name, e => {
          e.preventDefault();
          e.stopPropagation();
        }, false);
      });

      function bindDropzone(dzSelector, inpSelector, handler) {
        const dz = $(dzSelector);
        if (!dz) return;

        ['dragenter', 'dragover'].forEach(name => {
          dz.addEventListener(name, e => {
            e.preventDefault();
            e.stopPropagation();
            dz.classList.add('dragover');
          }, false);
        });

        ['dragleave', 'dragend'].forEach(name => {
          dz.addEventListener(name, e => {
            e.preventDefault();
            e.stopPropagation();
            dz.classList.remove('dragover');
          }, false);
        });

        dz.addEventListener('drop', e => {
          e.preventDefault();
          e.stopPropagation();
          dz.classList.remove('dragover');
          const dt = e.dataTransfer;
          if (dt && dt.files && dt.files.length > 0) {
            handler(dt.files[0]);
          }
        }, false);
      }

      bindDropzone('#i2i-dropzone', '#i2i-file', handleFileSelect);
      bindDropzone('#tagger-dropzone', '#tagger-file', handleTaggerUpload);

      // 2. Support Ctrl+V paste from clipboard directly into active tab
      window.addEventListener('paste', e => {
        const items = (e.clipboardData || e.originalEvent?.clipboardData)?.items;
        if (!items) return;
        for (const item of items) {
          if (item.kind === 'file' && item.type.startsWith('image/')) {
            const file = item.getAsFile();
            if (gState.activeTab === 'tagger') {
              handleTaggerUpload(file);
              showToast("Pasted image into Auto-Tagger!", "success");
            } else {
              switchTab('img2img');
              handleFileSelect(file);
              showToast("Pasted image into Img2Img!", "success");
            }
            break;
          }
        }
      });
    }

    window.onload = () => { loadModels(); initDragAndDrop(); };
  </script>
  <div id="toast-container" class="toast-container"></div>
</body>
</html>
'''

@app.get("/")
def index():
    return INDEX_HTML

if __name__ == "__main__":
    port = int(os.environ.get("APP_PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
