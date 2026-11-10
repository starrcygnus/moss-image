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
from flask import Flask, Response, jsonify, request

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
HERE = os.path.dirname(os.path.abspath(__file__))
COMFY_DIR = os.environ.get("COMFY_DIR", "/content/ComfyUI")
CLIENT_ID = uuid.uuid4().hex

JOB_PARAMS = {}

def embed_png_metadata(img_path, meta_info, final_caption):
    if not os.path.isfile(img_path):
        return
    try:
        from PIL import Image, PngImagePlugin
        with Image.open(img_path) as im:
            w, h = im.size
            im_format = im.format or "PNG"
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
            f"Steps: {p.get('steps', 25)}",
            f"Sampler: {p.get('sampler', 'er_sde')}",
            f"Schedule type: {p.get('scheduler', 'simple')}",
            f"CFG scale: {p.get('cfg', 2.5)}",
            f"Seed: {seed}",
            f"Size: {w}x{h}",
            f"Model: {p.get('checkpoint', 'moss_image')}",
        ]
        l1 = p.get("lora1")
        if l1 and l1 != "None":
            param_parts.append(f"LoRA 1: {l1} ({p.get('lora1_strength', 1.0)})")
        l2 = p.get("lora2")
        if l2 and l2 != "None":
            param_parts.append(f"LoRA 2: {l2} ({p.get('lora2_strength', 1.0)})")
        l3 = p.get("lora3")
        if l3 and l3 != "None":
            param_parts.append(f"LoRA 3: {l3} ({p.get('lora3_strength', 1.0)})")
        if p.get("upsampler"):
            param_parts.append("Prompt Upsampler: True")
            if raw_prompt and raw_prompt != prompt_text:
                param_parts.append(f"Original prompt: {raw_prompt}")
        if p.get("slg"):
            param_parts.append("Skip-Layer Guidance: True")
        if p.get("input_image"):
            c_mode = p.get("control_mode", "img2img")
            param_parts.append(f"Mode: {c_mode}")
            param_parts.append(f"Input image: {p.get('input_image')}")
            if c_mode == "control":
                param_parts.append(f"Control Type: {p.get('control_type', 'auto')}")
                param_parts.append(f"Control LoRA: {p.get('control_lora', 'None')} ({p.get('control_strength', 1.0)})")
            else:
                param_parts.append(f"Denoise: {p.get('denoise', 0.65)}")

        lines.append(", ".join(param_parts))
        a1111_params = "\n".join(lines)

        # Resolve dynamic links in ComfyUI prompt JSON so tools like sdimage.info see the text on CLIPTextEncode
        if "prompt" in existing_info:
            try:
                p_dict = json.loads(existing_info["prompt"])
                if "18:4" in p_dict and "inputs" in p_dict["18:4"]:
                    p_dict["18:4"]["inputs"]["text"] = prompt_text
                if "18:17" in p_dict and "inputs" in p_dict["18:17"]:
                    p_dict["18:17"]["inputs"]["value"] = prompt_text
                existing_info["prompt"] = json.dumps(p_dict)
            except Exception:
                pass

        def clean_latin1(text):
            if not text:
                return ""
            replacements = {
                '\u201c': '"', '\u201d': '"',
                '\u2018': "'", '\u2019': "'",
                '\u2014': '-', '\u2013': '-',
                '\u2026': '...', '\u00a0': ' ',
            }
            for orig, rep in replacements.items():
                text = text.replace(orig, rep)
            return text.encode('latin-1', 'replace').decode('latin-1')

        pnginfo = PngImagePlugin.PngInfo()
        for k, v in existing_info.items():
            if isinstance(v, str):
                pnginfo.add_text(k, v)

        pnginfo.add_text("parameters", clean_latin1(a1111_params))
        pnginfo.add_text("caption", clean_latin1(prompt_text))
        pnginfo.add_text("Description", clean_latin1(prompt_text))
        pnginfo.add_text("Comment", clean_latin1(prompt_text))
        if wf:
            try:
                wf_copy = json.loads(json.dumps(wf))
                if "18:4" in wf_copy and "inputs" in wf_copy["18:4"]:
                    wf_copy["18:4"]["inputs"]["text"] = prompt_text
                if "18:17" in wf_copy and "inputs" in wf_copy["18:17"]:
                    wf_copy["18:17"]["inputs"]["value"] = prompt_text
                pnginfo.add_text("workflow", json.dumps(wf_copy))
            except Exception:
                pnginfo.add_text("workflow", json.dumps(wf))

        try:
            exif = existing_exif if existing_exif is not None else im_copy.getexif()
            exif[0x010e] = prompt_text[:2000]
            exif[0x9286] = f"Prompt: {prompt_text}\nNegative: {negative}\nSeed: {seed}"
            im_copy.save(img_path, format=im_format, pnginfo=pnginfo, exif=exif)
        except Exception:
            im_copy.save(img_path, format=im_format, pnginfo=pnginfo)
    except Exception as e:
        print(f"Metadata embedding note: {e}")


with open(os.path.join(HERE, "workflow_api.json"), encoding="utf-8") as f:
    WORKFLOW = json.load(f)

DEFAULTS = {
    "prompt": WORKFLOW.get("18:17", {}).get("inputs", {}).get("value", ""),
    "negative": WORKFLOW.get("18:5", {}).get("inputs", {}).get("text", ""),
    "steps": WORKFLOW.get("18:7", {}).get("inputs", {}).get("steps", 25),
    "cfg": WORKFLOW.get("18:7", {}).get("inputs", {}).get("cfg", 2.5),
    "sampler": WORKFLOW.get("18:7", {}).get("inputs", {}).get("sampler_name", "er_sde"),
    "scheduler": WORKFLOW.get("18:7", {}).get("inputs", {}).get("scheduler", "simple"),
    "aspect_ratio": WORKFLOW.get("19", {}).get("inputs", {}).get("aspect_ratio", "1:1 (Square)"),
    "megapixels": WORKFLOW.get("19", {}).get("inputs", {}).get("megapixels", 1),
    "checkpoint": WORKFLOW.get("18:1", {}).get("inputs", {}).get("unet_name", "moss_image_sft_int8_convrot.safetensors"),
    "upsampler": bool(WORKFLOW.get("18:16", {}).get("inputs", {}).get("switch", True)),
    "slg": bool(WORKFLOW.get("18:24", {}).get("inputs", {}).get("switch", False)),
    "lora1": "None",
    "lora1_strength": 1.0,
    "lora2": "None",
    "lora2_strength": 1.0,
    "lora3": "None",
    "lora3_strength": 1.0,
    "control_mode": "img2img",
    "control_type": "auto",
    "control_lora": "None",
    "control_strength": 1.0,
}

app = Flask(__name__)

def comfy_get(path, timeout=15, **kw):
    r = requests.get(COMFY + path, timeout=timeout, **kw)
    r.raise_for_status()
    return r.json()

def combo(spec):
    if not spec:
        return []
    if isinstance(spec[0], list):
        return spec[0]
    if len(spec) > 1 and isinstance(spec[1], dict) and "options" in spec[1]:
        return list(spec[1]["options"])
    return []

def required_inputs(node_class):
    try:
        return comfy_get(f"/object_info/{node_class}")[node_class]["input"]["required"]
    except Exception:
        return {}

def clamp(value, lo, hi, cast, default):
    try:
        return max(lo, min(hi, cast(value)))
    except (TypeError, ValueError):
        return default


INPUT_DIR = os.path.join(COMFY_DIR, "input")
os.makedirs(INPUT_DIR, exist_ok=True)

def extract_metadata(file_path):
    meta = {}
    try:
        from PIL import Image
        with Image.open(file_path) as im:
            w, h = im.size
            meta["width"], meta["height"] = w, h
            info = dict(getattr(im, "info", {}))

            # 1. Automatic1111 'parameters' text chunk
            params = info.get("parameters")
            if params:
                p_str = str(params)
                parts = p_str.split("\nNegative prompt: ")
                if len(parts) > 1:
                    meta["prompt"] = parts[0].strip()
                    subparts = parts[1].split("\nSteps: ")
                    meta["negative"] = subparts[0].strip()
                    if len(subparts) > 1:
                        meta["params_line"] = "Steps: " + subparts[1].strip()
                else:
                    subparts = parts[0].split("\nSteps: ")
                    meta["prompt"] = subparts[0].strip()
                    if len(subparts) > 1:
                        meta["params_line"] = "Steps: " + subparts[1].strip()

                if "params_line" in meta:
                    m_seed = re.search(r"Seed:\s*(\d+)", meta["params_line"])
                    if m_seed:
                        meta["seed"] = int(m_seed.group(1))
                    m_cfg = re.search(r"CFG scale:\s*([\d\.]+)", meta["params_line"])
                    if m_cfg:
                        meta["cfg"] = float(m_cfg.group(1))
                    m_steps = re.search(r"Steps:\s*(\d+)", meta["params_line"])
                    if m_steps:
                        meta["steps"] = int(m_steps.group(1))

            # 2. ComfyUI 'prompt' JSON chunk
            elif "prompt" in info:
                try:
                    p_dict = json.loads(info["prompt"])
                    for nid, nval in p_dict.items():
                        c_type = nval.get("class_type")
                        if c_type == "CLIPTextEncode":
                            txt = nval.get("inputs", {}).get("text", "")
                            if isinstance(txt, str) and txt.strip():
                                if "negative" in str(nval.get("_meta", {}).get("title", "")).lower() or nid in ("18:5", "5"):
                                    meta["negative"] = txt.strip()
                                elif "prompt" not in meta:
                                    meta["prompt"] = txt.strip()
                        elif c_type == "PrimitiveStringMultiline":
                            val = nval.get("inputs", {}).get("value", "")
                            if isinstance(val, str) and val.strip() and "prompt" not in meta:
                                meta["prompt"] = val.strip()
                except Exception:
                    pass

            # 3. Fallback to caption / Description / Comment chunks
            if not meta.get("prompt"):
                for tag in ("caption", "Description", "Comment"):
                    val = info.get(tag)
                    if val and isinstance(val, str) and val.strip():
                        meta["prompt"] = val.strip()
                        break
    except Exception as e:
        print(f"Metadata extract note: {e}")
    return meta

def build_workflow(p):
    wf = copy.deepcopy(WORKFLOW)
    seed = clamp(p.get("seed", -1), -1, 2**50, int, -1)
    if seed < 0:
        seed = random.randint(0, 2**50)

    wf["18:17"]["inputs"]["value"] = (p.get("prompt") or "").strip()
    wf["18:5"]["inputs"]["text"] = (p.get("negative") or "").strip()

    ks = wf["18:7"]["inputs"]
    ks["seed"] = seed
    ks["steps"] = clamp(p.get("steps"), 1, 150, int, DEFAULTS["steps"])
    ks["cfg"] = clamp(p.get("cfg"), 0, 30, float, DEFAULTS["cfg"])
    ks["sampler_name"] = p.get("sampler") or DEFAULTS["sampler"]
    ks["scheduler"] = p.get("scheduler") or DEFAULTS["scheduler"]

    wf["19"]["inputs"]["aspect_ratio"] = p.get("aspect_ratio") or DEFAULTS["aspect_ratio"]
    wf["19"]["inputs"]["megapixels"] = clamp(p.get("megapixels"), 0.25, 4, float, 1.0)
    wf["18:16"]["inputs"]["switch"] = bool(p.get("upsampler", DEFAULTS["upsampler"]))
    wf["18:24"]["inputs"]["switch"] = bool(p.get("slg", DEFAULTS["slg"]))
    wf["18:11"]["inputs"]["sampling_mode.seed"] = seed
    wf["18:1"]["inputs"]["unet_name"] = p.get("checkpoint") or DEFAULTS["checkpoint"]

    curr_model = ["18:1", 0]
    curr_clip = ["18:2", 0]

    l1_name = p.get("lora1", "None")
    l1_str = float(p.get("lora1_strength", 1.0))
    is_turbo = "8-step" in (str(l1_name) + str(p.get("lora2", "")) + str(p.get("lora3", ""))).lower()
    # Turbo LoRA Protection: 8-step DMD LoRA requires CFG <= 1.5 and SLG off to avoid NaN latent explosion
    if is_turbo:
        if ks["cfg"] > 1.5:
            ks["cfg"] = 1.0
        wf["18:24"]["inputs"]["switch"] = False
    if l1_name and l1_name != "None" and l1_str != 0:
        wf["lora_1"] = {
            "inputs": {"lora_name": l1_name, "strength_model": l1_str, "strength_clip": l1_str, "model": curr_model, "clip": curr_clip},
            "class_type": "LoraLoader",
            "_meta": {"title": f"LoRA 1 ({l1_name})"}
        }
        curr_model = ["lora_1", 0]
        curr_clip = ["lora_1", 1]

    l2_name = p.get("lora2", "None")
    l2_str = float(p.get("lora2_strength", 1.0))
    if l2_name and l2_name != "None" and l2_str != 0:
        wf["lora_2"] = {
            "inputs": {"lora_name": l2_name, "strength_model": l2_str, "strength_clip": l2_str, "model": curr_model, "clip": curr_clip},
            "class_type": "LoraLoader",
            "_meta": {"title": f"LoRA 2 ({l2_name})"}
        }
        curr_model = ["lora_2", 0]
        curr_clip = ["lora_2", 1]

    l3_name = p.get("lora3", "None")
    l3_str = float(p.get("lora3_strength", 1.0))
    if l3_name and l3_name != "None" and l3_str != 0:
        wf["lora_3"] = {
            "inputs": {"lora_name": l3_name, "strength_model": l3_str, "strength_clip": l3_str, "model": curr_model, "clip": curr_clip},
            "class_type": "LoraLoader",
            "_meta": {"title": f"LoRA 3 ({l3_name})"}
        }
        curr_model = ["lora_3", 0]
        curr_clip = ["lora_3", 1]

    # Routing between Text-to-Image, Classic Img2Img, and Control LoRA
    active_tab = p.get("tab", "gen")
    input_img = (p.get("input_image") or "").strip()
    c_mode = p.get("control_mode", "img2img")

    if active_tab == "img2img" and input_img and c_mode == "control":
        # Control LoRA Mode (uses TextEncodeMOSSImageEdit)
        ctrl_lora = p.get("control_lora", "None")
        ctrl_str = float(p.get("control_strength", 1.0))
        ctrl_type = str(p.get("control_type", "auto")).lower()

        if ctrl_type == "auto":
            l_lower = ctrl_lora.lower()
            if any(k in l_lower for k in ("pose", "openpose", "dwpose")):
                ctrl_type = "pose"
            elif "depth" in l_lower:
                ctrl_type = "depth"
            elif any(k in l_lower for k in ("canny", "edge")):
                ctrl_type = "canny"
            else:
                ctrl_type = "direct"

        wf["input_load"] = {
            "inputs": {"image": input_img},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Reference Image"}
        }

        # Apply Preprocessor
        if ctrl_type == "pose":
            wf["ctrl_preprocess"] = {
                "inputs": {
                    "image": ["input_load", 0],
                    "detect_hand": "enable",
                    "detect_body": "enable",
                    "detect_face": "enable",
                    "resolution": 1024,
                    "bbox_detector": "yolox_l.onnx",
                    "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt"
                },
                "class_type": "DWPreprocessor",
                "_meta": {"title": "DWPose Skeleton Extractor"}
            }
            ctrl_img_link = ["ctrl_preprocess", 0]
        elif ctrl_type == "depth":
            wf["ctrl_preprocess"] = {
                "inputs": {
                    "image": ["input_load", 0],
                    "ckpt_name": "depth_anything_v2_vitl.pth",
                    "resolution": 1024
                },
                "class_type": "DepthAnythingV2Preprocessor",
                "_meta": {"title": "Depth Anything V2 Extractor"}
            }
            ctrl_img_link = ["ctrl_preprocess", 0]
        elif ctrl_type == "canny":
            wf["ctrl_preprocess"] = {
                "inputs": {
                    "image": ["input_load", 0],
                    "low_threshold": 0.4,
                    "high_threshold": 0.8
                },
                "class_type": "Canny",
                "_meta": {"title": "Canny Edge Extractor"}
            }
            ctrl_img_link = ["ctrl_preprocess", 0]
        else:
            ctrl_img_link = ["input_load", 0]

        # Attach Control LoRA to model chain
        if ctrl_lora and ctrl_lora != "None" and ctrl_str != 0:
            wf["lora_ctrl"] = {
                "inputs": {
                    "lora_name": ctrl_lora,
                    "strength_model": ctrl_str,
                    "strength_clip": ctrl_str,
                    "model": curr_model,
                    "clip": curr_clip
                },
                "class_type": "LoraLoader",
                "_meta": {"title": f"Control LoRA ({ctrl_lora})"}
            }
            curr_model = ["lora_ctrl", 0]
            curr_clip = ["lora_ctrl", 1]

        # Connect TextEncodeMOSSImageEdit for Positive and Negative conditionings
        prompt_input = ["18:16", 0] if wf["18:16"]["inputs"]["switch"] else (p.get("prompt") or "")
        wf["ctrl_encode_pos"] = {
            "inputs": {
                "prompt": prompt_input,
                "clip": curr_clip,
                "vae": ["18:3", 0],
                "edit_format": "ai-toolkit",
                "image_strength": 1.0,
                "image_0": ctrl_img_link
            },
            "class_type": "TextEncodeMOSSImageEdit",
            "_meta": {"title": "Text Encode MOSS-Image Edit (positive)"}
        }
        wf["ctrl_encode_neg"] = {
            "inputs": {
                "prompt": p.get("negative") or "",
                "clip": curr_clip,
                "vae": ["18:3", 0],
                "edit_format": "ai-toolkit",
                "image_strength": 1.0,
                "image_0": ctrl_img_link
            },
            "class_type": "TextEncodeMOSSImageEdit",
            "_meta": {"title": "Text Encode MOSS-Image Edit (negative)"}
        }

        # KSampler uses empty latent + 1.0 denoise for full generative structure
        ks["latent_image"] = ["18:6", 0]
        ks["denoise"] = 1.0
        ks["positive"] = ["ctrl_encode_pos", 0]
        ks["negative"] = ["ctrl_encode_neg", 0]
        if not is_turbo and ("cfg" not in p or p.get("cfg") == DEFAULTS["cfg"]):
            ks["cfg"] = 4.0

    elif active_tab == "img2img" and input_img:
        # Classic img2img Mode
        denoise_val = clamp(p.get("denoise", 0.65), 0.05, 1.0, float, 0.65)
        wf["input_load"] = {
            "inputs": {"image": input_img},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Input Image"}
        }
        wf["input_encode"] = {
            "inputs": {"pixels": ["input_load", 0], "vae": ["18:3", 0]},
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"}
        }
        ks["latent_image"] = ["input_encode", 0]
        ks["denoise"] = denoise_val
    else:
        # Pure Text-to-Image Mode
        ks["latent_image"] = ["18:6", 0]
        ks["denoise"] = 1.0

    if curr_model != ["18:1", 0]:
        wf["18:15"]["inputs"]["model"] = curr_model
        wf["18:24"]["inputs"]["on_false"] = curr_model
    if curr_clip != ["18:2", 0]:
        wf["18:4"]["inputs"]["clip"] = curr_clip
        wf["18:5"]["inputs"]["clip"] = curr_clip
        if "18:11" in wf and "clip" in wf["18:11"]["inputs"]:
            wf["18:11"]["inputs"]["clip"] = curr_clip

    return wf, seed

def explain_validation(resp):
    try:
        data = resp.json()
    except ValueError:
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

def find_error(entry):
    for kind, data in entry.get("status", {}).get("messages", []):
        if kind == "execution_error" and isinstance(data, dict):
            return f'{data.get("node_type", "node")}: {data.get("exception_message", "execution failed")}'.strip()
    return None

@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")

@app.get("/api/health")
def health():
    try:
        comfy_get("/system_stats", timeout=3)
        return jsonify(ok=True)
    except Exception:
        return jsonify(ok=False), 503

@app.get("/api/options")
def options():
    res = required_inputs("ResolutionSelector")
    ks = required_inputs("KSampler")
    un = required_inputs("UNETLoader")
    checkpoints = combo(un.get("unet_name"))
    for std_ckpt in ["moss_image_sft_int8_convrot.safetensors", "moss_image_base_int8_convrot.safetensors", "kromoss_int8_convrot.safetensors"]:
        if std_ckpt not in checkpoints:
            checkpoints.append(std_ckpt)

    for mdir_name in ["diffusion_models", "checkpoints", "unet"]:
        mdir = os.path.join(COMFY_DIR, "models", mdir_name)
        if os.path.exists(mdir):
            for root, _, files in os.walk(mdir):
                for f in files:
                    if f.endswith((".safetensors", ".pt", ".bin", ".ckpt")):
                        rel = os.path.relpath(os.path.join(root, f), mdir)
                        if rel not in checkpoints:
                            checkpoints.append(rel)

    lora_spec = required_inputs("LoraLoader")
    loras = ["None"] + [l for l in combo(lora_spec.get("lora_name")) if l != "None"]
    lora_dir = os.path.join(COMFY_DIR, "models", "loras")
    if os.path.exists(lora_dir):
        for root, _, files in os.walk(lora_dir):
            for f in files:
                if f.endswith((".safetensors", ".pt", ".bin", ".ckpt")):
                    rel = os.path.relpath(os.path.join(root, f), lora_dir)
                    if rel not in loras:
                        loras.append(rel)

    return jsonify(
        aspect_ratios=combo(res.get("aspect_ratio")) or [DEFAULTS["aspect_ratio"]],
        samplers=combo(ks.get("sampler_name")) or [DEFAULTS["sampler"]],
        schedulers=combo(ks.get("scheduler")) or [DEFAULTS["scheduler"]],
        checkpoints=checkpoints,
        loras=loras,
        defaults=DEFAULTS,
    )


@app.post("/api/upload")
def upload_image():
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify(error="Empty filename"), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        return jsonify(error="Unsupported format. Please upload PNG, JPG, or WEBP."), 400

    unique_name = f"ref_{uuid.uuid4().hex[:10]}{ext}"
    save_path = os.path.join(INPUT_DIR, unique_name)

    # Extract metadata first from raw upload stream
    f.seek(0)
    meta = extract_metadata(f)

    # Re-open with Pillow to scale & ensure dimensions are multiples of 16 (for FLUX/MOSS DiT RoPE)
    f.seek(0)
    from PIL import Image
    im = Image.open(f).convert("RGB")
    w, h = im.size
    if max(w, h) > 1536:
        scale = 1536 / max(w, h)
        w, h = int(w * scale), int(h * scale)
    w = max(256, (w // 16) * 16)
    h = max(256, (h // 16) * 16)
    if im.size != (w, h):
        im = im.resize((w, h), Image.Resampling.LANCZOS)
    im.save(save_path, format="PNG")

    meta["width"], meta["height"] = w, h
    return jsonify(filename=unique_name, url=f"/api/input_image?filename={unique_name}", metadata=meta)

@app.get("/api/input_image")
def get_input_image():
    fn = request.args.get("filename", "")
    safe = os.path.basename(fn)
    path = os.path.join(INPUT_DIR, safe)
    if not os.path.isfile(path):
        return "Not found", 404
    ext = os.path.splitext(safe)[1].lower()
    mimes = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    return Response(open(path, "rb").read(), mimetype=mimes.get(ext, "image/png"))

_florence_model = None
_florence_processor = None

def get_florence():
    global _florence_model, _florence_processor
    if _florence_model is None:
        import torch
        from transformers import AutoProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        loaded = False
        # Try native Florence2ForConditionalGeneration first (native in modern transformers)
        try:
            from transformers import Florence2ForConditionalGeneration
            print("Loading Florence2ForConditionalGeneration from florence-community/Florence-2-base...")
            _florence_model = Florence2ForConditionalGeneration.from_pretrained(
                "florence-community/Florence-2-base",
                torch_dtype=dtype
            ).to(device).eval()
            _florence_processor = AutoProcessor.from_pretrained("florence-community/Florence-2-base")
            loaded = True
            print("✓ Native Florence-2 loaded successfully.")
        except Exception as e:
            print(f"Native loader skipped ({e}), falling back to AutoModel...")

        if not loaded:
            from transformers import AutoConfig, AutoModelForCausalLM
            from unittest.mock import patch
            from transformers.dynamic_module_utils import get_imports

            def fixed_get_imports(filename):
                if not str(filename).endswith("modeling_florence2.py"):
                    return get_imports(filename)
                imports = get_imports(filename)
                if "flash_attn" in imports:
                    imports.remove("flash_attn")
                return imports

            for model_id in ["florence-community/Florence-2-base", "microsoft/Florence-2-base"]:
                try:
                    print(f"Loading {model_id} on {device} ({dtype})...")
                    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                    if hasattr(config, "text_config"):
                        setattr(config.text_config, "forced_bos_token_id", None)
                        try:
                            setattr(type(config.text_config), "forced_bos_token_id", None)
                        except Exception:
                            pass
                    try:
                        setattr(type(config), "forced_bos_token_id", None)
                    except Exception:
                        pass

                    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                        _florence_model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            config=config,
                            torch_dtype=dtype,
                            trust_remote_code=True
                        ).to(device).eval()
                        _florence_processor = AutoProcessor.from_pretrained(
                            model_id,
                            trust_remote_code=True
                        )
                    print(f"✓ {model_id} loaded successfully.")
                    break
                except Exception as err:
                    print(f"Failed loading {model_id}: {err}")

        # Ensure forced_bos_token_id is safely set to None on any config object
        if _florence_model is not None:
            if hasattr(_florence_model.config, "text_config"):
                setattr(_florence_model.config.text_config, "forced_bos_token_id", None)
                try:
                    setattr(type(_florence_model.config.text_config), "forced_bos_token_id", None)
                except Exception:
                    pass
            if hasattr(_florence_model, "generation_config") and _florence_model.generation_config is not None:
                setattr(_florence_model.generation_config, "forced_bos_token_id", None)

    return _florence_model, _florence_processor

@app.post("/api/describe")
def describe_image():
    p = request.get_json(force=True, silent=True) or {}
    fn = p.get("filename")
    if not fn:
        return jsonify(error="No image specified"), 400
    safe = os.path.basename(fn)
    path = os.path.join(INPUT_DIR, safe)
    if not os.path.isfile(path):
        return jsonify(error="Reference image not found on server"), 404

    try:
        from PIL import Image
        import torch

        im = Image.open(path).convert("RGB")
        model, processor = get_florence()
        task = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=task, images=im, return_tensors="pt")

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        inputs = {k: (v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device)) for k, v in inputs.items()}

        if hasattr(model.config, "text_config"):
            try:
                type(model.config.text_config).forced_bos_token_id = None
            except Exception:
                pass
            model.config.text_config.forced_bos_token_id = None
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.forced_bos_token_id = None

        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                forced_bos_token_id=None
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(generated_text, task=task, image_size=im.size)

        caption = ""
        if isinstance(parsed, dict):
            caption = parsed.get(task, "")
            if not caption and len(parsed) > 0:
                caption = list(parsed.values())[0]
        elif isinstance(parsed, str):
            caption = parsed

        caption = (caption or "").strip()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify(caption=caption)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=f"Reverse prompt error: {str(e)}"), 500


@app.post("/api/generate")
def generate():
    try:
        p = request.get_json(force=True, silent=True) or {}
        if not (p.get("prompt") or "").strip():
            return jsonify(error="Please enter a prompt."), 400
        wf, seed = build_workflow(p)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=f"Workflow build error: {str(e)}"), 500
    try:
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
        return jsonify(error=f"Backend is not reachable: {e}"), 502
    if r.status_code != 200:
        return jsonify(error=explain_validation(r)), 400
    prompt_id = r.json()["prompt_id"]
    JOB_PARAMS[prompt_id] = {"params": p, "seed": seed, "workflow": wf}
    return jsonify(prompt_id=prompt_id, seed=seed)

@app.get("/api/status/<pid>")
def status(pid):
    try:
        hist = comfy_get(f"/history/{pid}")
    except Exception as e:
        return jsonify(state="error", error=f"Backend is not reachable: {e}"), 502

    if pid in hist:
        entry = hist[pid]
        outputs = entry.get("outputs", {})
        images = outputs.get("9", {}).get("images", [])
        if images:
            caption = ""
            preview = outputs.get("18:13", {}).get("text")
            if preview:
                caption = preview[0] if isinstance(preview, list) else str(preview)
            meta_info = JOB_PARAMS.get(pid, {})
            for img in images:
                subfolder = img.get("subfolder", "")
                filename = img.get("filename", "")
                out_dir = os.path.join(COMFY_DIR, "output", subfolder) if subfolder else os.path.join(COMFY_DIR, "output")
                img_path = os.path.join(out_dir, filename)
                embed_png_metadata(img_path, meta_info, caption)
            return jsonify(state="done", images=images, caption=caption)
        err = find_error(entry)
        if err:
            return jsonify(state="error", error=err)
        if entry.get("status", {}).get("completed"):
            return jsonify(state="error", error="The workflow finished but produced no image.")

    try:
        q = comfy_get("/queue")
    except Exception:
        return jsonify(state="queued", position=None)
    if any(len(i) > 1 and i[1] == pid for i in q.get("queue_running", [])):
        return jsonify(state="running")
    pending = [i[1] for i in q.get("queue_pending", []) if len(i) > 1]
    if pid in pending:
        return jsonify(state="queued", position=pending.index(pid) + 1)
    return jsonify(state="running")

@app.post("/api/cancel")
def cancel():
    pid = (request.get_json(silent=True) or {}).get("prompt_id")
    if not pid:
        return jsonify(ok=False), 400
    try:
        q = comfy_get("/queue")
        if any(len(i) > 1 and i[1] == pid for i in q.get("queue_running", [])):
            requests.post(f"{COMFY}/interrupt", timeout=10)
        else:
            requests.post(f"{COMFY}/queue", json={"delete": [pid]}, timeout=10)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502
    return jsonify(ok=True)


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
    cmd = [
        "aria2c", "-x", "8", "-s", "8", "-k", "1M", "-c", "--file-allocation=none",
        "--console-log-level=warn", "--summary-interval=1",
        f"--user-agent={UA}",
        "--header=Referer: https://civitai.com/",
        "--check-certificate=false",
        "-d", dest_dir, "-o", fname, final_url
    ]
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
        req_headers = {"User-Agent": UA, "Referer": "https://civitai.com/"}
        if headers:
            req_headers.update(headers)
        with requests.get(final_url, headers=req_headers, stream=True, timeout=60) as r:
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

def run_download_task(raw_entry, target_type, civitai_key):
    global DOWNLOAD_STATE
    try:
        url = raw_entry.strip()
        custom_name = None
        if "|" in url and not url.startswith(("http://", "https://", "urn:")):
            custom_name, url = [p.strip() for p in url.split("|", 1)]

        m_air = re.match(r"urn:air:(?P<eco>[^:]+):(?P<type>[^:]+):(?P<src>[^:]+):(?P<model_id>\d+)(?:@(?P<version_id>\d+))?(?:\+(?P<file_id>\d+))?", url)
        if m_air:
            air_type = m_air.group("type").lower()
            if "checkpoint" in air_type or "model" in air_type:
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
        if target_type == "auto":
            if "civitai" in domain:
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
                if any(k in url.lower() for k in ["checkpoint", "diffusion_models", "unet", "moss_image", "base"]):
                    target_type = "checkpoint"
                else:
                    target_type = "lora"

        dest_dir = os.path.join(COMFY_DIR, "models", "diffusion_models" if target_type == "checkpoint" else "loras")
        os.makedirs(dest_dir, exist_ok=True)
        label = "Checkpoint" if target_type == "checkpoint" else "LoRA"

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
                fname = f"civitai_{label.lower()}_{version_id or 'custom'}.safetensors"
            fname = ensure_valid_ext(fname)

            DOWNLOAD_STATE["filename"] = fname
            DOWNLOAD_STATE["msg"] = f"Downloading {fname}..."
            download_file_robust(final_url, dest_dir, fname, headers)

        else:
            if "huggingface.co" in domain and "/blob/" in url:
                url = url.replace("/blob/", "/resolve/", 1)
            fname = custom_name or urllib.parse.unquote(url.split("?")[0].rstrip("/").split("/")[-1])
            fname = ensure_valid_ext(fname)
            DOWNLOAD_STATE["filename"] = fname
            DOWNLOAD_STATE["msg"] = f"Downloading {fname}..."
            download_file_robust(url, dest_dir, fname)

        DOWNLOAD_STATE["percent"] = 100
        DOWNLOAD_STATE["status"] = "done"
        DOWNLOAD_STATE["msg"] = f"✓ {fname} ready! Added to {label}s."
        DOWNLOAD_STATE["filename"] = fname
    except Exception as e:
        import traceback
        traceback.print_exc()
        DOWNLOAD_STATE["status"] = "error"
        DOWNLOAD_STATE["error"] = str(e)
        DOWNLOAD_STATE["msg"] = f"Download failed: {str(e)}"
    finally:
        DOWNLOAD_STATE["active"] = False

@app.post("/api/download_model")
def api_download_model():
    global DOWNLOAD_STATE
    p = request.get_json(force=True, silent=True) or {}
    url = (p.get("url") or "").strip()
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
            "target_type": p.get("type", "auto")
        }

    target_type = p.get("type", "auto")
    api_key = (p.get("api_key") or "").strip()

    t = threading.Thread(target=run_download_task, args=(url, target_type, api_key), daemon=True)
    t.start()
    return jsonify(ok=True, message="Download started")

@app.get("/api/download_status")
def api_download_status():
    return jsonify(DOWNLOAD_STATE)

@app.get("/api/image")
def image():
    params = {"filename": request.args.get("filename", ""), "subfolder": request.args.get("subfolder", ""), "type": request.args.get("type", "output")}
    if params["type"] not in ("output", "temp") or not params["filename"]:
        return "bad request", 400
    subfolder = params["subfolder"]
    filename = params["filename"]
    seed = request.args.get("seed", "")
    base, ext = os.path.splitext(filename)
    if not ext:
        ext = ".png"
    dl_filename = f"{base}_seed{seed}{ext}" if seed and f"seed{seed}" not in base else filename
    out_dir = os.path.join(COMFY_DIR, "output", subfolder) if subfolder else os.path.join(COMFY_DIR, "output")
    local_path = os.path.join(out_dir, filename)
    if os.path.isfile(local_path):
        with open(local_path, "rb") as f:
            content = f.read()
        return Response(content, mimetype="image/png", headers={
            "Cache-Control": "public, max-age=3600",
            "Content-Disposition": f'inline; filename="{dl_filename}"'
        })
    r = requests.get(f"{COMFY}/view", params=params, timeout=60)
    if r.status_code != 200:
        return "not found", 404
    return Response(r.content, mimetype=r.headers.get("Content-Type", "image/png"), headers={
        "Cache-Control": "public, max-age=3600",
        "Content-Disposition": f'inline; filename="{dl_filename}"'
    })

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MOSS-Image & Kromoss</title>
<style>
:root{--bg:#0e0f13;--panel:#171922;--line:#262a38;--txt:#e8eaf0;--mut:#8b91a5;--acc:#7c8cff;--acc2:#a78bfa;--err:#ff6b6b}
*{box-sizing:border-box}html,body{margin:0;padding:0;background:var(--bg);color:var(--txt);font:15px/1.45 system-ui,-apple-system,sans-serif;overflow-x:hidden;max-width:100vw}
header{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--line)}
header h1{font-size:18px;margin:0;font-weight:650}#conn{margin-left:auto;font-size:12px;color:var(--mut)}
#conn i{display:inline-block;width:8px;height:8px;border-radius:50%;background:#555;margin-right:6px}
#conn.ok i{background:#3ddc97}#conn.bad i{background:var(--err)}
main{display:grid;grid-template-columns:40% minmax(0,1fr);gap:14px;padding:12px 14px;max-width:100vw;box-sizing:border-box;margin:0}
@media(max-width:800px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;min-width:0;box-sizing:border-box}
label{display:block;font-size:12px;color:var(--mut);margin:12px 0 5px;text-transform:uppercase;letter-spacing:.5px}
label:first-child{margin-top:0}
textarea,input[type=number],select{width:100%;background:#0f1117;color:var(--txt);border:1px solid var(--line);border-radius:9px;padding:9px 10px;font:inherit}
textarea{resize:vertical}input[type=range]{width:100%;accent-color:var(--acc)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.chk{display:flex;gap:9px;align-items:flex-start;font-size:14px;margin:12px 0 0}.chk input{margin-top:3px;accent-color:var(--acc)}
.chk small{display:block;color:var(--mut);font-size:12px}
.lora-box{background:#11131a;border:1px solid var(--line);border-radius:10px;padding:10px;margin-top:12px}
.tab-bar{display:flex;gap:6px;margin:0 0 10px 0;border-bottom:1px solid var(--line);padding-bottom:8px}
.tab-btn{flex:1;background:transparent;color:var(--mut);border:1px solid transparent;border-radius:8px;padding:8px 10px;font-size:12px;font-weight:600;cursor:pointer;transition:all .15s ease}
.tab-btn:hover{color:var(--txt);background:rgba(255,255,255,0.04)}
.tab-btn.active{color:#fff;background:linear-gradient(135deg,rgba(124,140,255,0.25),rgba(167,139,250,0.25));border-color:var(--acc)}
.btns{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.btns-top{position:sticky;top:0;z-index:20;background:var(--panel);padding:4px 0 14px 0;margin:0 0 12px 0;border-bottom:1px solid var(--line)}#go{flex:1;min-width:120px;padding:11px 16px;font-size:14px;box-shadow:0 4px 14px rgba(124,140,255,0.3);letter-spacing:.3px}#stop{padding:11px 14px;white-space:nowrap}.batch-box{display:flex;align-items:center;gap:6px;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:0 12px}.batch-box label{font-size:11px;color:var(--mut);text-transform:uppercase;font-weight:600;margin:0;letter-spacing:.5px}.batch-box input{width:46px;border:0;background:transparent;padding:10px 2px;text-align:center;font-weight:700;color:var(--txt);font-size:15px}.batch-box input:focus{outline:none}.dropzone{border:2px dashed var(--line);border-radius:12px;padding:12px;text-align:center;cursor:pointer;background:var(--bg);transition:border-color .2s,background .2s;margin-top:10px}.dropzone:hover,.dropzone.dragover{border-color:var(--acc);background:rgba(124,140,255,0.06)}.dropzone-empty{display:flex;align-items:center;justify-content:center;gap:10px;color:var(--mut);font-size:13px}.dropzone-empty b{color:var(--txt)}.dropzone-preview{display:flex;align-items:center;gap:12px}.dropzone-preview img{width:56px;height:56px;object-fit:cover;border-radius:8px;border:1px solid var(--line)}.dropzone-info{flex:1;text-align:left;overflow:hidden}.dropzone-info .fn{font-weight:600;font-size:13px;color:var(--txt);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.meta-badge{display:inline-block;font-size:11px;background:rgba(124,140,255,0.18);color:var(--acc);padding:2px 8px;border-radius:6px;margin-top:3px;font-weight:600}.drop-actions{display:flex;gap:6px;align-items:center}.btn-describe{background:linear-gradient(135deg,rgba(124,140,255,0.22),rgba(160,120,255,0.22));border:1px solid var(--acc);color:var(--txt);font-size:12px;font-weight:600;padding:5px 10px;border-radius:6px;cursor:pointer;white-space:nowrap;transition:all .15s ease}.btn-describe:hover{background:var(--acc);color:#fff;box-shadow:0 2px 8px rgba(124,140,255,0.4)}.btn-describe:disabled{opacity:0.5;cursor:not-allowed}.ghost-sm{background:transparent;border:1px solid var(--line);color:var(--mut);font-size:12px;padding:4px 10px;border-radius:6px;cursor:pointer}.ghost-sm:hover{color:var(--err);border-color:var(--err)}
button{border:0;border-radius:10px;padding:11px 16px;font:inherit;font-weight:600;cursor:pointer;color:#fff;background:linear-gradient(135deg,var(--acc),var(--acc2))}
button:disabled{opacity:.5;cursor:not-allowed}button.ghost{background:transparent;border:1px solid var(--line);color:var(--txt)}
#go{flex:1}
#stage{min-height:300px;display:flex;align-items:center;justify-content:center;background:#0f1117;border:1px dashed var(--line);border-radius:12px;overflow:hidden}
#stage img{max-width:100%;max-height:calc(100vh - 230px);object-fit:contain;display:block}#placeholder{color:var(--mut)}#status{margin:12px 2px 0;color:var(--mut);min-height:20px}
#status.err{color:var(--err)}#actions{display:flex;gap:10px;margin-top:10px;align-items:center;flex-wrap:wrap}#actions a{color:var(--acc);text-decoration:none;font-weight:600}
details{margin-top:12px;background:#0f1117;border:1px solid var(--line);border-radius:10px;padding:8px 12px}summary{cursor:pointer;color:var(--mut)}
pre{white-space:pre-wrap;word-break:break-word;margin:10px 0 2px;font:13px/1.5 ui-monospace,Menlo,monospace}
h3{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px;margin:20px 0 8px}
#hist{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px;width:100%;max-width:100%;min-width:0;box-sizing:border-box}
#hist::-webkit-scrollbar{height:6px}#hist::-webkit-scrollbar-thumb{background:var(--line);border-radius:3px}
#hist img{width:64px;height:64px;min-width:64px;aspect-ratio:1;object-fit:cover;border-radius:8px;border:1px solid var(--line);cursor:pointer;transition:transform .1s,border-color .1s}
#hist img:hover{transform:scale(1.05);border-color:var(--acc)}[hidden]{display:none!important}
</style>
</head>
<body>
<header><h1>MOSS-Image & Kromoss</h1><span id="conn"><i></i>connecting…</span></header>
<main>
  <section class="panel">
    <div class="btns btns-top"><button id="go">✨ Generate</button><div class="batch-box"><label for="batch">Images</label><input type="number" id="batch" min="1" max="10" value="1" title="How many images to generate (e.g. 2 for 2 variations)"></div><button id="stop" class="ghost" hidden>Cancel</button></div>
    
    <div class="tab-bar">
      <button type="button" class="tab-btn active" id="tab-btn-gen">🎨 Text-to-Image</button>
      <button type="button" class="tab-btn" id="tab-btn-img2img">🖼️ Img2Img & Control</button>
      <button type="button" class="tab-btn" id="tab-btn-dl">📥 Downloader</button>
    </div>

    <!-- TAB 1: TEXT-TO-IMAGE -->
    <div id="tab-gen">
      <label for="prompt">Prompt</label><textarea id="prompt" rows="4" placeholder="Describe the image you want to generate..."></textarea>
      <label for="negative">Negative prompt</label><textarea id="negative" rows="1" placeholder="worst quality, bad anatomy..."></textarea>
      <div class="row" style="margin-top:8px">
        <div><label for="seed">Seed</label><input type="number" id="seed" min="0"></div>
        <label class="chk" style="margin-top:22px"><input type="checkbox" id="rand" checked><span>Random</span></label>
      </div>
      <details id="advanced-settings" style="margin-top:10px">
        <summary style="font-weight:600;font-size:12px;color:var(--txt)">⚙️ Model & Sampler Settings</summary>
        <div style="margin-top:8px">
          <div class="row">
            <div><label for="aspect">Aspect</label><select id="aspect"></select></div>
            <div><label for="ckpt">Checkpoint</label><select id="ckpt"></select></div>
          </div>
          <div class="lora-box">
            <div class="row">
              <div><label for="lora1">LoRA 1</label><select id="lora1"></select></div>
              <div><label for="lora1_str">Strength</label><input type="number" id="lora1_str" min="-2" max="2" step="0.05" value="1.0"></div>
            </div>
            <div class="row" style="margin-top:6px">
              <div><label for="lora2">LoRA 2</label><select id="lora2"></select></div>
              <div><label for="lora2_str">Strength</label><input type="number" id="lora2_str" min="-2" max="2" step="0.05" value="1.0"></div>
            </div>
            <div class="row" style="margin-top:6px">
              <div><label for="lora3">LoRA 3</label><select id="lora3"></select></div>
              <div><label for="lora3_str">Strength</label><input type="number" id="lora3_str" min="-2" max="2" step="0.05" value="1.0"></div>
            </div>
          </div>
          <label for="mp" style="margin-top:8px">Megapixels: <b id="mpv"></b></label><input type="range" id="mp" min="0.25" max="4" step="0.05">
          <div class="row">
            <div><label for="steps">Steps</label><input type="number" id="steps" min="1" max="150"></div>
            <div><label for="cfg">CFG</label><input type="number" id="cfg" min="0" max="30" step="0.1"></div>
          </div>
          <div class="row">
            <div><label for="sampler">Sampler</label><select id="sampler"></select></div>
            <div><label for="scheduler">Scheduler</label><select id="scheduler"></select></div>
          </div>
          <label class="chk" style="margin-top:8px"><input type="checkbox" id="upsampler"><span>Prompt upsampler<small>Expand captions</small></span></label>
          <label class="chk"><input type="checkbox" id="slg"><span>Skip-layer guidance<small>Layer 22 details</small></span></label>
        </div>
      </details>
      <label class="chk" style="margin-top:10px;font-size:12px"><input type="checkbox" id="notif" checked><span>🔔 Chime alert <a href="#" id="test-sound" style="color:var(--acc);text-decoration:underline;margin-left:4px;">(test)</a></span></label>
    </div>

    <!-- TAB 2: IMG2IMG & CONTROL -->
    <div id="tab-img2img" style="display:none">
      <div class="dropzone" id="dropzone" style="margin-top:0">
        <input type="file" id="file-input" accept="image/png,image/jpeg,image/webp" hidden>
        <div id="drop-empty" class="dropzone-empty">
          <span style="font-size:22px">🖼️</span>
          <div><b>Drop Reference Image or Control Map</b><br><small style="font-size:11px;color:var(--mut)">Supports photos, pose skeletons, depth maps, or sketches</small></div>
        </div>
        <div id="drop-filled" class="dropzone-preview" hidden>
          <img id="drop-thumb" alt="Preview">
          <div class="dropzone-info">
            <div class="fn" id="drop-fname">photo.png</div>
            <div id="drop-meta-badge" class="meta-badge" hidden>✨ Embedded prompt loaded</div>
          </div>
          <div class="drop-actions">
            <button type="button" id="drop-describe" class="btn-describe" title="Analyze image and generate prompt with Florence-2">🔍 AI Describe</button>
            <button type="button" id="drop-clear" class="ghost-sm" title="Remove reference image">✕ Remove</button>
          </div>
        </div>
      </div>

      <!-- Mode Selector -->
      <div style="margin-top:10px;background:#11131a;border:1px solid var(--line);border-radius:10px;padding:10px">
        <div style="font-size:11px;font-weight:700;color:var(--txt);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">Image Guidance Mode</div>
        <div style="display:flex;gap:8px">
          <button type="button" id="mode-btn-i2i" class="tab-btn active" style="flex:1;border:1px solid var(--acc)">🔄 Classic img2img</button>
          <button type="button" id="mode-btn-ctrl" class="tab-btn" style="flex:1;border:1px solid var(--line)">🎛️ ControlNet (LoRA)</button>
        </div>

        <!-- Classic img2img Controls -->
        <div id="panel-mode-i2i" style="margin-top:10px">
          <label for="denoise" style="margin-top:0">Denoise: <b id="denoise-val">0.65</b> <small style="display:inline;margin-left:6px;font-size:10px;color:var(--mut)">(0.35=keep source, 0.70=creative)</small></label>
          <input type="range" id="denoise" min="0.10" max="1.0" step="0.05" value="0.65">
        </div>

        <!-- ControlNet Controls -->
        <div id="panel-mode-ctrl" style="margin-top:10px;display:none">
          <div class="row">
            <div>
              <label for="ctrl-type" style="margin-top:0">Preprocessor</label>
              <select id="ctrl-type" style="font-size:12px;padding:8px">
                <option value="auto">Auto (Detect from LoRA)</option>
                <option value="pose">🦴 Pose (DWPose Skeleton)</option>
                <option value="depth">🏔️ Depth (Depth Anything V2)</option>
                <option value="canny">🖋️ Canny (Edge Outlines)</option>
                <option value="direct">🎯 Direct (Already Control Map)</option>
              </select>
            </div>
            <div>
              <label for="ctrl-str" style="margin-top:0">Control Strength</label>
              <input type="number" id="ctrl-str" min="-2" max="2" step="0.05" value="1.0" style="font-size:12px;padding:8px">
            </div>
          </div>
          <div style="margin-top:8px">
            <label for="ctrl-lora">Control LoRA</label>
            <select id="ctrl-lora" style="font-size:12px;padding:8px"></select>
          </div>
          <div id="ctrl-hint" style="font-size:11px;color:var(--acc);margin-top:6px;line-height:1.4">
            💡 Auto: auto-detects pose, depth, or canny from the selected Control LoRA.
          </div>
        </div>
      </div>

      <label for="prompt-i2i" style="margin-top:10px">Prompt</label><textarea id="prompt-i2i" rows="3" placeholder="Describe how you want to transform or style the image..."></textarea>
      <label for="negative-i2i">Negative prompt</label><textarea id="negative-i2i" rows="1" placeholder="worst quality, bad anatomy..."></textarea>

      <div class="row" style="margin-top:8px">
        <div><label for="seed-i2i">Seed</label><input type="number" id="seed-i2i" min="0"></div>
        <label class="chk" style="margin-top:22px"><input type="checkbox" id="rand-i2i" checked><span>Random</span></label>
      </div>
      <div style="margin-top:10px;font-size:11px;color:var(--mut)">
        ℹ️ Uses the Checkpoint, LoRA, and Sampler settings configured in the drawer.
      </div>
    </div>

    <!-- TAB 2: DOWNLOADER -->
    <div id="tab-dl" style="display:none">
      <div style="background:#11131a;border:1px solid var(--line);border-radius:10px;padding:12px">
        <div style="font-weight:700;font-size:13px;color:var(--txt);margin-bottom:4px">📥 Download Models & LoRAs</div>
        <div style="font-size:11px;color:var(--mut);margin-bottom:12px">Download extra diffusion models or LoRAs directly from Civitai, Hugging Face, or Google Drive into ComfyUI.</div>

        <label for="dl-url" style="margin-top:0">Model / LoRA URL or Civitai AIR</label>
        <input type="text" id="dl-url" placeholder="https://civitai.com/... or urn:air:... or huggingface.co/..." style="font-size:12px;padding:8px">

        <div class="row" style="margin-top:8px">
          <div>
            <label for="dl-type">Save as</label>
            <select id="dl-type" style="font-size:12px;padding:8px">
              <option value="auto">Auto-Detect</option>
              <option value="lora">LoRA</option>
              <option value="checkpoint">Checkpoint / Model</option>
            </select>
          </div>
          <div>
            <label for="dl-key">Civitai Key (optional)</label>
            <input type="password" id="dl-key" placeholder="API key if restricted" style="font-size:12px;padding:8px">
          </div>
        </div>

        <div style="margin-top:12px">
          <button type="button" id="dl-btn" style="width:100%;padding:10px;font-size:13px;background:linear-gradient(135deg,var(--acc),var(--acc2));border-radius:8px">⬇️ Download to WebUI</button>
        </div>

        <div id="dl-status" style="margin-top:8px;font-size:12px;color:var(--mut);min-height:20px;word-break:break-all;line-height:1.4"></div>
      </div>

      <div style="margin-top:12px;background:rgba(255,255,255,0.02);border:1px solid var(--line);border-radius:8px;padding:10px;font-size:11px;color:var(--mut);line-height:1.5">
        <b style="color:var(--txt)">💡 Supported formats:</b>
        <ul style="margin:6px 0 0;padding-left:16px">
          <li><b>Civitai:</b> Model page URL (e.g. <code>civitai.com/models/...</code>) or mirror <code>civitai.red</code></li>
          <li><b>Civitai AIR:</b> <code>urn:air:mossimage:checkpoint:...</code></li>
          <li><b>Hugging Face:</b> Direct model links (auto-converts <code>/blob/</code>)</li>
          <li><b>Google Drive:</b> Shared file or folder links</li>
        </ul>
      </div>
    </div>
  </section>
  <section class="panel">
    <div id="stage"><span id="placeholder">Your image will appear here</span></div>
    <div id="status"></div>
    <div id="actions" hidden><a id="dl" download>Download</a><button id="reuse" class="ghost">Reuse this seed</button></div>
    <details id="capt" hidden><summary>Final prompt sent to the model</summary><pre id="captxt"></pre></details>
    <h3>History</h3><div id="hist"></div>
  </section>
</main>
<script>
const $ = s => document.querySelector(s);
const HKEY = "moss_history_v1";
let pid = null, tStart = 0, polling = false, history = [], lastSeed = null;
function fill(sel, items, value){
  const el = $(sel); el.innerHTML = "";
  items.forEach(v => { const o = document.createElement("option"); o.value = o.textContent = v; el.appendChild(o); });
  if (value !== undefined) el.value = value;
}
function setSettings(s){
  $("#prompt").value = s.prompt || "";
  if ($("#prompt-i2i")) $("#prompt-i2i").value = s.prompt || "";
  $("#negative").value = s.negative || "";
  if ($("#negative-i2i")) $("#negative-i2i").value = s.negative || "";
  $("#aspect").value = s.aspect_ratio; $("#ckpt").value = s.checkpoint;
  $("#lora1").value = s.lora1 || "None"; $("#lora1_str").value = s.lora1_strength ?? 1.0;
  $("#lora2").value = s.lora2 || "None"; $("#lora2_str").value = s.lora2_strength ?? 1.0;
  $("#lora3").value = s.lora3 || "None"; $("#lora3_str").value = s.lora3_strength ?? 1.0;
  if ($("#ctrl-lora")) $("#ctrl-lora").value = s.control_lora || "None";
  if ($("#ctrl-str")) $("#ctrl-str").value = s.control_strength ?? 1.0;
  if ($("#ctrl-type")) $("#ctrl-type").value = s.control_type || "auto";
  $("#mp").value = s.megapixels; $("#mpv").textContent = (+s.megapixels).toFixed(2);
  $("#steps").value = s.steps; $("#cfg").value = s.cfg;
  $("#sampler").value = s.sampler; $("#scheduler").value = s.scheduler;
  $("#upsampler").checked = !!s.upsampler; $("#slg").checked = !!s.slg;
}
function getSettings(){
  const isI2I = window._activeTab === "img2img";
  const pVal = (isI2I && $("#prompt-i2i") && $("#prompt-i2i").value.trim()) ? $("#prompt-i2i").value : $("#prompt").value;
  const nVal = (isI2I && $("#negative-i2i") && $("#negative-i2i").value.trim()) ? $("#negative-i2i").value : $("#negative").value;
  const isRand = isI2I ? ($("#rand-i2i") ? $("#rand-i2i").checked : true) : $("#rand").checked;
  const sEl = isI2I ? $("#seed-i2i") : $("#seed");
  const sVal = isRand ? -1 : parseInt((sEl ? sEl.value : "") || "0", 10);

  return {
    tab: window._activeTab || "gen",
    prompt: pVal, negative: nVal,
    aspect_ratio: $("#aspect").value, checkpoint: $("#ckpt").value,
    lora1: $("#lora1").value, lora1_strength: parseFloat($("#lora1_str").value) || 0,
    lora2: $("#lora2").value, lora2_strength: parseFloat($("#lora2_str").value) || 0,
    lora3: $("#lora3").value, lora3_strength: parseFloat($("#lora3_str").value) || 0,
    megapixels: parseFloat($("#mp").value), steps: parseInt($("#steps").value, 10),
    cfg: parseFloat($("#cfg").value), sampler: $("#sampler").value, scheduler: $("#scheduler").value,
    seed: sVal,
    upsampler: $("#upsampler").checked, slg: $("#slg").checked,
    input_image: isI2I ? (window._inputImage || null) : null,
    control_mode: window._controlMode || "img2img",
    control_type: $("#ctrl-type") ? $("#ctrl-type").value : "auto",
    control_lora: $("#ctrl-lora") ? $("#ctrl-lora").value : "None",
    control_strength: parseFloat($("#ctrl-str") ? $("#ctrl-str").value : "1.0") || 1.0,
    denoise: parseFloat($("#denoise") ? $("#denoise").value : "0.65") || 0.65
  };
}
function say(msg, isErr){ const s = $("#status"); s.textContent = msg; s.className = isErr ? "err" : ""; }
function busy(b){ $("#go").disabled = b; $("#stop").hidden = !b; }
function imgUrl(i, seed){
  let u = "/api/image?filename=" + encodeURIComponent(i.filename) + "&subfolder=" + encodeURIComponent(i.subfolder || "") + "&type=" + encodeURIComponent(i.type || "output");
  if (seed !== undefined && seed !== null && seed !== "") u += "&seed=" + encodeURIComponent(seed);
  return u;
}
function show(url, caption, seed, filename){
  const img = new Image();
  img.onload = () => { say("Done \u2022 " + img.naturalWidth + "\u00d7" + img.naturalHeight + " \u2022 seed " + seed); };
  img.src = url; $("#stage").innerHTML = ""; $("#stage").appendChild(img);
  $("#dl").href = url;
  if (filename) {
    const dot = filename.lastIndexOf(".");
    const base = dot !== -1 ? filename.slice(0, dot) : filename;
    const ext = dot !== -1 ? filename.slice(dot) : ".png";
    const dlName = (seed !== undefined && seed !== null && !base.includes(String(seed))) ? `${base}_seed${seed}${ext}` : filename;
    $("#dl").setAttribute("download", dlName);
  }
  $("#actions").hidden = false;
  $("#captxt").textContent = caption || ""; $("#capt").hidden = !caption; lastSeed = seed;
}
function saveHistory(){ try { localStorage.setItem(HKEY, JSON.stringify(history.slice(0, 24))); } catch(e){} }
function loadHistory(){ try { history = JSON.parse(localStorage.getItem(HKEY) || "[]"); } catch(e){ history = []; } }
function renderHistory(){
  const box = $("#hist"); box.innerHTML = "";
  history.forEach(h => {
    const t = new Image(); t.src = h.url; t.loading = "lazy"; t.title = h.settings.prompt.slice(0, 120);
    t.onerror = () => t.remove();
    t.onclick = () => {
      setSettings(h.settings);
      $("#rand").checked = false;
      const sVal = (h.settings && h.settings.seed !== undefined) ? h.settings.seed : h.seed;
      $("#seed").value = sVal;
      show(h.url, h.caption, sVal, h.filename);
    };
    box.appendChild(t);
  });
}
let audioCtx = null;
function getAudioContext() {
  try {
    if (!audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) audioCtx = new AC();
    }
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    return audioCtx;
  } catch(e) { return null; }
}

function playFinishSound() {
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    const now = ctx.currentTime;
    const playTone = (freq, start, duration, vol) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, start);
      gain.gain.setValueAtTime(vol, start);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(start);
      osc.stop(start + duration);
    };
    // Harmonic 3-tone chime: D5 (587Hz) -> A5 (880Hz) -> D6 (1175Hz)
    playTone(587.33, now, 0.35, 0.25);
    playTone(880.00, now + 0.12, 0.55, 0.30);
    playTone(1174.66, now + 0.24, 0.70, 0.20);
  } catch(e) { console.warn("Audio error:", e); }
}

function sendFinishNotification(caption, seed) {
  if ("Notification" in window) {
    if (Notification.permission === "granted") {
      try {
        new Notification("MOSS-Image: Generation Done! 🎨", {
          body: (caption || "Your image is ready!").slice(0, 100) + (seed ? " (Seed: " + seed + ")" : ""),
          icon: "/favicon.ico"
        });
      } catch(e){}
    }
  }
}

let origTitle = document.title || "MOSS-Image Web UI";
let flashInterval = null;
function flashTitle(msg) {
  if (flashInterval) clearInterval(flashInterval);
  let state = false, count = 0;
  flashInterval = setInterval(() => {
    document.title = (state = !state) ? msg : origTitle;
    count++;
    if (count >= 16) {
      clearInterval(flashInterval);
      document.title = origTitle;
    }
  }, 750);
}
window.addEventListener("focus", () => {
  if (flashInterval) {
    clearInterval(flashInterval);
    document.title = origTitle;
  }
});

let cancelRequested = false;

async function pollOnce(id, prefix){
  let fails = 0;
  while (pid === id && !cancelRequested) {
    try {
      const r = await fetch("/api/status/" + id);
      if (!r.ok) throw new Error("HTTP " + r.status);
      const d = await r.json();
      fails = 0;
      const secs = Math.round((Date.now() - tStart) / 1000);
      if (d.state === "done") {
        const s = Object.assign({}, window._lastSettings);
        const url = imgUrl(d.images[0], s.seed);
        const fname = (d.images && d.images[0] && d.images[0].filename) ? d.images[0].filename : "";
        show(url, d.caption, s.seed, fname);
        history.unshift({url, caption: d.caption, settings: s, filename: fname, seed: s.seed}); saveHistory(); renderHistory();
        return { ok: true, caption: d.caption, seed: s.seed };
      }
      if (d.state === "error") { say((prefix || "") + (d.error || "Generation failed."), true); return { ok: false }; }
      if (d.state === "queued") say((prefix || "") + "Queued" + (d.position ? " (position " + d.position + ")" : "") + " • " + secs + "s");
      else say((prefix || "") + "Generating… " + secs + "s  (first run loads models and takes longer)");
    } catch (err) {
      fails++;
      if (fails >= 6) { say((prefix || "") + "Lost connection to the server: " + err, true); return { ok: false }; }
      say((prefix || "") + "Tunnel reconnecting (" + fails + "/6)… model inference still running");
    }
    await new Promise(r => setTimeout(r, 2000));
  }
  return { ok: false };
}

async function generate(){
  getAudioContext();
  if ($("#notif") && $("#notif").checked && "Notification" in window && Notification.permission === "default") {
    Notification.requestPermission();
  }
  const s = getSettings();
  if (s.tab === "img2img" && !s.input_image) { say("⚠️ Please drop or select a reference image for Img2Img / Control mode!", true); return; }
  if (!s.prompt.trim()) { say("Please enter a prompt.", true); return; }

  const count = Math.max(1, Math.min(10, parseInt($("#batch") ? $("#batch").value : "1", 10) || 1));
  busy(true);
  cancelRequested = false;

  const isRand = $("#rand").checked;
  let baseSeed = s.seed;
  let completed = 0;
  let lastCaption = "";

  for (let i = 0; i < count; i++) {
    if (cancelRequested) break;

    const currentSeed = isRand ? -1 : (baseSeed >= 0 ? baseSeed + i : -1);
    s.seed = currentSeed;
    const prefix = count > 1 ? `[${i + 1}/${count}] ` : "";

    say(prefix + "Submitting…");
    try {
      const r = await fetch("/api/generate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(s)});
      let d = {};
      try {
        d = await r.json();
      } catch (err) {
        say((prefix || "") + `Backend error (${r.status}): Server returned non-JSON response.`, true);
        break;
      }
      if (!r.ok) { say((prefix || "") + (d.error || "Request failed."), true); break; }
      s.seed = d.seed; window._lastSettings = Object.assign({}, s);
      if (isRand) $("#seed").value = d.seed;
      pid = d.prompt_id; tStart = Date.now();
      const res = await pollOnce(pid, prefix);
      if (!res.ok) break;
      completed++;
      lastCaption = res.caption || "";
    } catch (e) {
      say((prefix || "") + "Could not reach the server: " + e, true);
      break;
    }
  }

  pid = null;
  busy(false);

  if (completed > 0 && !cancelRequested && $("#notif") && $("#notif").checked) {
    playFinishSound();
    sendFinishNotification(count > 1 ? `Generated ${completed} of ${count} images!` : lastCaption, s.seed);
    flashTitle("🔔 Finished! • MOSS-Image");
  }
}

async function cancel(){
  cancelRequested = true;
  if (!pid) return; const id = pid; pid = null;
  try { await fetch("/api/cancel", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({prompt_id: id})}); } catch(e){}
  busy(false); say("Cancelled.");
}
async function health(){
  try {
    const r = await fetch("/api/health"); const el = $("#conn");
    el.className = r.ok ? "ok" : "bad"; el.innerHTML = "<i></i>" + (r.ok ? "Backend online" : "Backend offline");
  } catch(e){ const el = $("#conn"); el.className = "bad"; el.innerHTML = "<i></i>server unreachable"; }
}
async function init(){
  loadHistory();
  const o = await (await fetch("/api/options")).json();
  const d = o.defaults;
  fill("#aspect", o.aspect_ratios, d.aspect_ratio); fill("#ckpt", o.checkpoints, d.checkpoint);
  fill("#lora1", o.loras, d.lora1); fill("#lora2", o.loras, d.lora2); fill("#lora3", o.loras, d.lora3);
  fill("#ctrl-lora", o.loras, d.control_lora || "None");
  if ($("#ctrl-lora").value === "None" && o.loras) {
    const found = o.loras.find(l => /control|pose|depth|canny/i.test(l));
    if (found) $("#ctrl-lora").value = found;
  }
  const checkTurbo = e => {
    if (e.target.value && e.target.value.toLowerCase().includes("8-step")) {
      $("#steps").value = 8;
      $("#cfg").value = 1.0;
      $("#slg").checked = false;
      say("⚡ Turbo LoRA active: Steps set to 8, CFG to 1.0, SLG turned off.");
    }
  };
  $("#lora1").addEventListener("change", checkTurbo);
  $("#lora2").addEventListener("change", checkTurbo);
  $("#lora3").addEventListener("change", checkTurbo);
  fill("#sampler", o.samplers, d.sampler); fill("#scheduler", o.schedulers, d.scheduler);
  setSettings(d);
  renderHistory(); health(); setInterval(health, 15000);
  const notifEl = $("#notif");
  if (notifEl) {
    const saved = localStorage.getItem("moss_notif");
    if (saved !== null) notifEl.checked = saved === "1";
    notifEl.addEventListener("change", () => {
      localStorage.setItem("moss_notif", notifEl.checked ? "1" : "0");
      if (notifEl.checked) getAudioContext();
    });
  }
  const testBtn = $("#test-sound");
  if (testBtn) {
    testBtn.addEventListener("click", e => {
      e.preventDefault(); e.stopPropagation();
      getAudioContext();
      playFinishSound();
      if ("Notification" in window) {
        if (Notification.permission === "default") {
          Notification.requestPermission().then(p => {
            if (p === "granted") new Notification("MOSS-Image 🎨", {body: "Sound & desktop notifications active!"});
          });
        } else if (Notification.permission === "granted") {
          new Notification("MOSS-Image 🎨", {body: "Sound & desktop notifications active!"});
        }
      }
    });
  }
}
$("#mp").addEventListener("input", e => $("#mpv").textContent = (+e.target.value).toFixed(2));
  window._activeTab = "gen";
  window._controlMode = "img2img";

  const tabGenBtn = $("#tab-btn-gen");
  const tabI2iBtn = $("#tab-btn-img2img");
  const tabDlBtn = $("#tab-btn-dl");
  const tabGen = $("#tab-gen");
  const tabI2i = $("#tab-img2img");
  const tabDl = $("#tab-dl");

  function switchTab(t) {
    window._activeTab = t;
    if (tabGenBtn) tabGenBtn.classList.toggle("active", t === "gen");
    if (tabI2iBtn) tabI2iBtn.classList.toggle("active", t === "img2img");
    if (tabDlBtn) tabDlBtn.classList.toggle("active", t === "dl");
    if (tabGen) tabGen.style.display = t === "gen" ? "block" : "none";
    if (tabI2i) tabI2i.style.display = t === "img2img" ? "block" : "none";
    if (tabDl) tabDl.style.display = t === "dl" ? "block" : "none";
  }

  if (tabGenBtn) tabGenBtn.addEventListener("click", () => switchTab("gen"));
  if (tabI2iBtn) tabI2iBtn.addEventListener("click", () => switchTab("img2img"));
  if (tabDlBtn) tabDlBtn.addEventListener("click", () => switchTab("dl"));

  // Sync prompts across tabs
  $("#prompt").addEventListener("input", e => { if ($("#prompt-i2i")) $("#prompt-i2i").value = e.target.value; });
  if ($("#prompt-i2i")) $("#prompt-i2i").addEventListener("input", e => { $("#prompt").value = e.target.value; });
  $("#negative").addEventListener("input", e => { if ($("#negative-i2i")) $("#negative-i2i").value = e.target.value; });
  if ($("#negative-i2i")) $("#negative-i2i").addEventListener("input", e => { $("#negative").value = e.target.value; });

  // Control mode toggle
  if ($("#mode-btn-i2i")) {
    $("#mode-btn-i2i").addEventListener("click", () => {
      window._controlMode = "img2img";
      $("#mode-btn-i2i").classList.add("active");
      $("#mode-btn-i2i").style.borderColor = "var(--acc)";
      $("#mode-btn-ctrl").classList.remove("active");
      $("#mode-btn-ctrl").style.borderColor = "var(--line)";
      $("#panel-mode-i2i").style.display = "block";
      $("#panel-mode-ctrl").style.display = "none";
    });
  }
  if ($("#mode-btn-ctrl")) {
    $("#mode-btn-ctrl").addEventListener("click", () => {
      window._controlMode = "control";
      $("#mode-btn-ctrl").classList.add("active");
      $("#mode-btn-ctrl").style.borderColor = "var(--acc)";
      $("#mode-btn-i2i").classList.remove("active");
      $("#mode-btn-i2i").style.borderColor = "var(--line)";
      $("#panel-mode-ctrl").style.display = "block";
      $("#panel-mode-i2i").style.display = "none";
    });
  }

  const ctrlHints = {
    auto: "💡 Auto: auto-detects pose, depth, or canny from the selected Control LoRA.",
    pose: "🦴 Pose: Upload any photo of a person. DWPose will auto-extract the skeleton.",
    depth: "🏔️ Depth: Upload any scene or object. Depth Anything will compute 3D distances.",
    canny: "🖋️ Canny: Upload any image. Canny will detect clean structural outlines.",
    direct: "🎯 Direct: Upload a pre-made black & white skeleton, depth map, or edge drawing."
  };
  if ($("#ctrl-type")) {
    $("#ctrl-type").addEventListener("change", e => {
      $("#ctrl-hint").textContent = ctrlHints[e.target.value] || ctrlHints.auto;
    });
  }

  const dlBtn = $("#dl-btn");
  const dlUrlInput = $("#dl-url");
  const dlTypeSelect = $("#dl-type");
  const dlKeyInput = $("#dl-key");
  const dlStatus = $("#dl-status");
  let dlPollTimer = null;

  async function pollDownload() {
    try {
      const res = await fetch("/api/download_status");
      const st = await res.json();
      if (!st) return;

      if (st.status === "downloading") {
        dlBtn.disabled = true;
        dlStatus.style.color = "var(--acc)";
        dlStatus.textContent = `⏳ [${st.percent || 0}%] ${st.msg || "Downloading..."}`;
        dlPollTimer = setTimeout(pollDownload, 1000);
      } else if (st.status === "done") {
        dlBtn.disabled = false;
        dlStatus.style.color = "#3ddc97";
        dlStatus.textContent = st.msg;
        say(st.msg);
        if (typeof playSuccessChime === "function") playSuccessChime();

        fetch("/api/options")
          .then(r => r.json())
          .then(opts => {
            if (opts.checkpoints) {
              const curCkpt = $("#ckpt").value;
              fill("#ckpt", opts.checkpoints, st.target_type === "checkpoint" ? st.filename : curCkpt);
            }
            if (opts.loras) {
              const curL1 = $("#lora1").value;
              const curL2 = $("#lora2").value;
              const curL3 = $("#lora3").value;
              const curCtrl = $("#ctrl-lora") ? $("#ctrl-lora").value : "None";
              fill("#lora1", opts.loras, st.target_type === "lora" ? st.filename : curL1);
              fill("#lora2", opts.loras, curL2);
              fill("#lora3", opts.loras, curL3);
              if ($("#ctrl-lora")) fill("#ctrl-lora", opts.loras, st.target_type === "lora" && /control|pose|depth|canny/i.test(st.filename) ? st.filename : curCtrl);
            }
            say(`🎉 New ${st.target_type || "model"} loaded and ready to use!`);
          });
      } else if (st.status === "error") {
        dlBtn.disabled = false;
        dlStatus.style.color = "var(--err)";
        dlStatus.textContent = "✗ " + (st.error || st.msg);
        say("Download failed: " + (st.error || st.msg), true);
      } else {
        dlBtn.disabled = false;
      }
    } catch (e) {
      dlBtn.disabled = false;
      dlStatus.textContent = "Error checking download status: " + e;
    }
  }

  if (dlBtn) {
    dlBtn.addEventListener("click", async () => {
      const url = (dlUrlInput.value || "").trim();
      if (!url) {
        dlStatus.style.color = "var(--err)";
        dlStatus.textContent = "Please paste a URL or Civitai AIR identifier.";
        return;
      }
      dlBtn.disabled = true;
      dlStatus.style.color = "var(--acc)";
      dlStatus.textContent = "Sending download request...";
      try {
        const res = await fetch("/api/download_model", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url: url,
            type: dlTypeSelect.value,
            api_key: dlKeyInput.value.trim()
          })
        });
        const data = await res.json();
        if (data.error) {
          dlBtn.disabled = false;
          dlStatus.style.color = "var(--err)";
          dlStatus.textContent = "✗ " + data.error;
          say(data.error, true);
        } else {
          pollDownload();
        }
      } catch (err) {
        dlBtn.disabled = false;
        dlStatus.style.color = "var(--err)";
        dlStatus.textContent = "Failed to start download: " + err;
      }
    });
  }

  $("#go").onclick = generate; $("#stop").onclick = cancel;
$("#reuse").onclick = () => { if (lastSeed !== null) { $("#rand").checked = false; $("#seed").value = lastSeed; } };
$("#prompt").addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") generate(); });

  window._inputImage = null;
  const dz = $("#dropzone");
  const fi = $("#file-input");
  const de = $("#drop-empty");
  const df = $("#drop-filled");
  const thumb = $("#drop-thumb");
  const fnEl = $("#drop-fname");
  const mbEl = $("#drop-meta-badge");
  const clrBtn = $("#drop-clear");
  const dnRow = $("#denoise-row");
  const dnInput = $("#denoise");
  const dnVal = $("#denoise-val");

  if (dnInput && dnVal) {
    dnInput.addEventListener("input", e => dnVal.textContent = (+e.target.value).toFixed(2));
  }

  function handleUpload(file) {
    if (!file) return;
    say("Uploading image & checking embedded metadata…");
    const fd = new FormData();
    fd.append("file", file);
    fetch("/api/upload", { method: "POST", body: fd })
      .then(r => r.json())
      .then(data => {
        if (data.error) { say(data.error, true); return; }
        window._inputImage = data.filename;
        thumb.src = data.url;
        fnEl.textContent = file.name;
        de.hidden = true;
        df.hidden = false;
        if (dnRow) dnRow.style.display = "block";

        const m = data.metadata || {};
        let loadedTags = [];
        if (m.prompt) {
          $("#prompt").value = m.prompt;
          if ($("#prompt-i2i")) $("#prompt-i2i").value = m.prompt;
          loadedTags.push("prompt");
        }
        if (m.negative) {
          $("#negative").value = m.negative;
          if ($("#negative-i2i")) $("#negative-i2i").value = m.negative;
          loadedTags.push("negative prompt");
        }
        if (m.seed !== undefined && m.seed !== null) {
          $("#seed").value = m.seed;
          $("#rand").checked = false;
          loadedTags.push("seed");
        }
        if (loadedTags.length > 0) {
          mbEl.hidden = false;
          mbEl.textContent = "✨ Loaded " + loadedTags.join(", ");
          say("📋 Extracted embedded " + loadedTags.join(" & ") + " from image!");
        } else {
          mbEl.hidden = true;
          say("Image ready for img2img! (Tip: Click '🔍 AI Describe' to reverse prompt with Florence-2)");
        }
      })
      .catch(e => say("Upload failed: " + e, true));
  }

  if (dz && fi) {
    dz.addEventListener("click", e => { if (e.target !== clrBtn && !clrBtn.contains(e.target)) fi.click(); });
    fi.addEventListener("change", e => { if (e.target.files && e.target.files[0]) handleUpload(e.target.files[0]); });
    dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("dragover"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", e => {
      e.preventDefault();
      dz.classList.remove("dragover");
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleUpload(e.dataTransfer.files[0]);
      }
    });
  }

  if (clrBtn) {
    clrBtn.addEventListener("click", e => {
      e.preventDefault();
      e.stopPropagation();
      window._inputImage = null;
      if (fi) fi.value = "";
      de.hidden = false;
      df.hidden = true;
      mbEl.hidden = true;
      if (dnRow) dnRow.style.display = "none";
      say("Reference image removed. Mode: Text-to-Image.");
    });

  const descBtn = $("#drop-describe");
  if (descBtn) {
    descBtn.addEventListener("click", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!window._inputImage) return;
      descBtn.disabled = true;
      const origText = descBtn.textContent;
      descBtn.textContent = "⏳ Analyzing...";
      say("Analyzing image with Florence-2 AI captioner (downloading ~460MB weights on first use)...");
      try {
        const res = await fetch("/api/describe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: window._inputImage })
        });
        const data = await res.json();
        if (data.error) {
          say("Florence-2 error: " + data.error, true);
        } else if (data.caption) {
          $("#prompt").value = data.caption;
          if ($("#prompt-i2i")) $("#prompt-i2i").value = data.caption;
          mbEl.hidden = false;
          mbEl.textContent = "🔍 Florence-2 caption generated";
          say("✨ AI description generated and copied into Prompt!");
          if (typeof playSuccessChime === "function") playSuccessChime();
        } else {
          say("No caption returned from model.", true);
        }
      } catch (err) {
        say("Reverse prompt request failed: " + err, true);
      } finally {
        descBtn.disabled = false;
        descBtn.textContent = origText;
      }
    });
  }
  }

init();
</script>
</body>
</html>"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), threaded=True)
