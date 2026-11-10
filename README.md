# 🖼️ MOSS-Image & Kromoss WebUI (Google Colab)

An interactive, zero-scroll web application for running **MOSS-Image** and **Kromoss** on Google Colab (Tesla T4 GPU).

[![Open in Colab (v12)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/starrcygnus/moss-image/blob/main/Moss_Image_v12.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/starrcygnus/moss-image/blob/main/Moss_Image.ipynb)

> 💡 **Credits:** Originally adapted from the MOSS-Image Colab notebook created by **[@AICHUCKY](https://github.com/AICHUCKY)** ([YouTube: @AIWithChucky](https://youtube.com/@AIWithChucky)).

---

### ✨ Features:
- 🎛️ **Dedicated Img2Img & ControlNet Tab (v12):** Separate workflow tab for image guidance.
  - **Classic img2img:** Denoise slider (`0.10`–`1.00`) to transform existing photos.
  - **ControlNet Mode (Control LoRA):** Native structural conditioning via MOSS-Image Edit encoders. Supports **Pose** (DWPose skeleton), **Depth** (Depth Anything V2), **Canny** (edge outlines), and **Direct** control maps.
  - **Auto-Preprocessors:** Drop any regular photo and the preprocessor auto-extracts the skeleton or depth map before generating.
- 🎨 **Pure Text-to-Image Tab:** Keeps pure prompt-to-image creation completely clean and uncluttered.
- 📥 **In-App Model & LoRA Downloader:** Download extra models, checkpoints, and LoRAs directly inside the running WebUI without stopping or restarting! Supports Civitai URLs, Civitai AIR identifiers (`urn:air:...`), Hugging Face, Google Drive, and direct links with live speed/percentage progress.
- 🎛️ **Triple LoRA Stacking:** Load up to 3 LoRAs simultaneously (e.g. Turbo LoRA + Style LoRA + Character LoRA) with individual strength controls and automatic 8-step DMD Turbo safeguards.
- 🔍 **AI Reverse Prompt (Florence-2):** Visual prompt generator powered by `microsoft/Florence-2-base` (~460 MB) with zero-OOM memory safety.
- 🖥️ **Widescreen Compact Layout (40% / 60%):** Designed specifically for laptop and small displays to eliminate vertical scrolling.
- 💾 **Automatic Seed Suffix:** Downloaded images automatically append the generation seed (e.g. `ComfyUI_00012_seed508518815257715.png`).
- 🌐 **Multi-Provider Tunnel Selector:** Cloudflare (HTTP/2 with no timeouts), Pinggy (SSH), ngrok, zrok, and Colab Native URL with automatic fallback.
- 🔢 **Batched Generation:** Generate 1 to 10 variations sequentially without GPU memory spikes.
- 🔔 **Chimes & Notifications:** Audio chime, desktop notification, and tab flashing on completion.

---

### 🙏 Acknowledgments & Credits
- **Original Notebook & Workflow:** **[AICHUCKY](https://github.com/AICHUCKY)** ([AI With Chucky on YouTube](https://youtube.com/@AIWithChucky))
- **MOSS-Image Model:** [OpenMOSS Team](https://github.com/OpenMOSS/MOSS-Image)
- **MOSS-Image Control LoRAs (Pose, Depth, Canny):** **[thedeoxen](https://huggingface.co/thedeoxen)** ([Hugging Face](https://huggingface.co/thedeoxen))
