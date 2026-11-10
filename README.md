# 🖼️ MOSS-Image & Kromoss WebUI (Google Colab)

An interactive, zero-scroll web application for running **MOSS-Image** and **Kromoss** on Google Colab (Tesla T4 GPU).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/starrcygnus/moss-image/blob/main/Moss_Image_v11.ipynb)

---

### ✨ Features:
- 📥 **In-App Model & LoRA Downloader:** Download extra models, checkpoints, and LoRAs directly inside the running WebUI without stopping or restarting! Supports Civitai URLs, Civitai AIR identifiers (`urn:air:...`), Hugging Face, Google Drive, and direct links with live speed/percentage progress.
- 🔍 **AI Reverse Prompt (Florence-2):** Visual prompt generator powered by `microsoft/Florence-2-base` (~460 MB) with zero-OOM memory safety.
- 🎨 **Smart img2img Workflow:** Dynamic reference dropzone, auto-scaling for FLUX/MOSS RoPE DiT embeddings, denoise slider (`0.10`–`1.00`), and automatic embedded prompt/seed extraction (A1111/ComfyUI/EXIF).
- 🖥️ **Widescreen Compact Layout (40% / 60%):** Designed specifically for laptop and small displays to eliminate vertical scrolling.
- 💾 **Automatic Seed Suffix:** Downloaded images automatically append the generation seed (e.g. `ComfyUI_00012_seed508518815257715.png`).
- 🌐 **Multi-Provider Tunnel Selector:** Cloudflare (HTTP/2 with no timeouts), Pinggy (SSH), ngrok, zrok, and Colab Native URL with automatic fallback.
- 🔢 **Batched Generation:** Generate 1 to 10 variations sequentially without GPU memory spikes.
- 🔔 **Chimes & Notifications:** Audio chime, desktop notification, and tab flashing on completion.
