import base64
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
import torch
from PIL import Image

from veto.editing.backends.flux2 import Flux2Editing
from veto.protection.config import DoubleStreamConfig, SingleStreamConfig
from veto.protection.engine import PerturbationEngine
from veto.protection.objectives.veto import VETO
from veto.protection.wrappers.factory import build_wrapper
from veto.utils.images import pil_to_tensor01, resize_image, tensor01_to_pil

_SHARED_WRAPPER = None
_SHARED_EDITOR = None


def get_shared_models(image_size: int = 512):
    global _SHARED_WRAPPER, _SHARED_EDITOR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _SHARED_WRAPPER is None:
        print("[VETO Demo] Pre-loading FLUX.2 pipeline into GPU VRAM...")
        _SHARED_WRAPPER = build_wrapper(
            model="flux2",
            device=device,
            image_size=image_size,
        )
        _SHARED_EDITOR = Flux2Editing(
            model_id="diffusers/FLUX.2-dev-bnb-4bit",
            device=device,
            num_inference_steps=28,
            guidance_scale=4.0,
            image_size=image_size,
            pipeline=_SHARED_WRAPPER.pipe,
        )
        print("[VETO Demo] FLUX.2 pipeline loaded successfully!")
    return _SHARED_WRAPPER, _SHARED_EDITOR


def resize_hw(image: Image.Image, height: int, width: int) -> Image.Image:
    image = image.convert("RGB")
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def run_protection_pipeline(
    input_image: Optional[Image.Image],
    epsilon: float,
    steps: int,
    step_size: float,
    height: int,
    width: int,
    inference_steps: int,
    guidance_scale: float,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[Image.Image], str, Optional[Image.Image], Optional[Image.Image]]:
    if input_image is None:
        return None, "❌ Please upload an image to protect.", None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height, width = int(height), int(width)

    try:
        # 1. Preprocess input image to (width, height)
        pil_img = resize_hw(input_image, height, width)
        x_source = pil_to_tensor01(pil_img).to(device)

        # 2. Get Shared Wrapper (already loaded in GPU)
        wrapper, _ = get_shared_models(max(height, width))

        double_stream = DoubleStreamConfig(
            enabled=True,
            layer_indices=[0],
            entropy_slices=["canvas_reference", "reference_canvas"],
        )
        single_stream = SingleStreamConfig(enabled=False, layer_indices=[], entropy_slices=[])

        objective = VETO(
            wrapper=wrapper,
            surrogate_prompt="",
            inference_steps=int(inference_steps),
            num_timesteps_per_step=int(inference_steps),
            guidance_scale=float(guidance_scale),
            double_stream=double_stream,
            single_stream=single_stream,
            seed=42,
        )

        # 3. Create Perturbation Engine with user-chosen Epsilon Budget
        engine = PerturbationEngine(
            objective=objective,
            eps=float(epsilon) / 255.0,
            alpha=float(step_size) / 255.0,
            steps=int(steps),
            momentum_decay=0.9,
            constraint_type="default_epsilon",
            seed=42,
        )

        # 4. Run PGD Optimization
        x_protected, delta, _ = engine.protect(x_source)
        protected_pil = tensor01_to_pil(x_protected)

        status_str = f"✅ Protection complete! Applied ε-ball bound of ε = {epsilon:.1f}/255 over {steps} PGD steps at {width}x{height} resolution."
        return protected_pil, status_str, protected_pil, pil_img
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            err_msg = (
                f"❌ CUDA Out of Memory Error: Resolution ({width}x{height}) exceeds available GPU memory.\n"
                f"💡 Solution: Please reduce Height and Width and run protection again."
            )
            return None, err_msg, None, None
        raise e


def run_editing_pipeline(
    clean_image: Optional[Image.Image],
    protected_image: Optional[Image.Image],
    edit_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    progress=gr.Progress(),
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    if clean_image is None or protected_image is None:
        return None, None, "❌ Please generate a VETO protected image in Step 1 first."
    if not edit_prompt or not edit_prompt.strip():
        return None, None, "❌ Please enter an editing prompt."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height, width = int(height), int(width)

    try:
        _, editor = get_shared_models(max(height, width))
        editor.num_inference_steps = int(num_inference_steps)
        editor.guidance_scale = float(guidance_scale)
        editor.image_size = max(height, width)

        clean_pil = resize_hw(clean_image, height, width)
        prot_pil = resize_hw(protected_image, height, width)

        # 1. Edit Clean Image
        progress(0.1, desc="Editing...")
        gen_clean = torch.Generator(device=device).manual_seed(int(seed))
        clean_edited = editor.edit_image(clean_pil, edit_prompt.strip(), generator=gen_clean)

        # 2. Edit VETO Protected Image
        progress(0.55, desc="Editing...")
        gen_prot = torch.Generator(device=device).manual_seed(int(seed))
        protected_edited = editor.edit_image(prot_pil, edit_prompt.strip(), generator=gen_prot)

        progress(1.0, desc="Editing comparison complete!")
        status_str = f"✅ Editing complete using prompt: \"{edit_prompt.strip()}\" at {width}x{height}\n"
        return clean_edited, protected_edited, status_str
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            err_msg = (
                f"❌ CUDA Out of Memory Error during FLUX.2 editing at {width}x{height}.\n"
                f"💡 Solution: Please reduce resolution in Phase 1 (e.g. to 768x768) and re-run protection."
            )
            return None, None, err_msg
        raise e


def update_dims_from_image(img: Optional[Image.Image]) -> Tuple[int, int, float]:
    if img is None:
        return 512, 512, 1.0
    w, h = img.size
    aspect = w / float(h)
    # Snap height and width to nearest multiple of 64 within [256, 1024]
    h_snap = max(256, min(1024, int(round(h / 64.0) * 64)))
    w_snap = max(256, min(1024, int(round(w / 64.0) * 64)))
    return h_snap, w_snap, aspect


def on_height_slider_release(height: int, lock_aspect: bool, aspect: float):
    if not lock_aspect or aspect <= 0:
        return gr.update()
    new_w = max(256, min(1024, int(round((height * aspect) / 64.0) * 64)))
    return gr.update(value=new_w)


def on_width_slider_release(width: int, lock_aspect: bool, aspect: float):
    if not lock_aspect or aspect <= 0:
        return gr.update()
    new_h = max(256, min(1024, int(round((width / aspect) / 64.0) * 64)))
    return gr.update(value=new_h)


# ── Logo base64 helper (works on any host, no file-serving needed) ──
def _logo_uri(filename: str) -> str:
    """Return a base64 data URI for a logo file in assets/logos/."""
    path = ASSETS_DIR / "logos" / filename
    if not path.exists():
        return ""  # graceful fallback
    ext = path.suffix.lstrip(".")
    mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


# Professional design system — paper green (#1A702B / #9AFF9A) on clean light backgrounds
CUSTOM_CSS = """
/* ── 1. CSS variable overrides — kill every dark-navy Gradio token ── */
:root,
.gradio-container,
[data-theme] {
    --color-background-primary: #FFFFFF;
    --color-background-secondary: #F8FAFC;
    --color-background-tertiary: #F1F5F9;
    --color-border-primary: #E2E8F0;
    --color-border-secondary: #CBD5E1;
    --color-accent: #1A702B;
    --color-accent-soft: #E6F4EA;
    --body-background-fill: #F3F7F4;
    --body-text-color: #0F172A;
    --body-text-color-subdued: #475569;
    --border-color-primary: #E2E8F0;
    --block-background-fill: #FFFFFF;
    --block-border-color: #E2E8F0;
    --block-border-width: 1px;
    --block-label-background-fill: #E6F4EA;
    --block-label-border-color: #C3E6CB;
    --block-label-text-color: #1A702B;
    --block-title-background-fill: #E6F4EA;
    --block-title-border-color: #C3E6CB;
    --block-title-text-color: #1A702B;
    --block-shadow: 0 1px 4px rgba(0,0,0,0.04);
    --block-radius: 10px;
    --checkbox-background-color: #FFFFFF;
    --checkbox-background-color-focus: #F0FBF4;
    --checkbox-background-color-hover: #F0FBF4;
    --checkbox-background-color-selected: #1A702B;
    --checkbox-border-color: #CBD5E1;
    --checkbox-border-color-focus: #1A702B;
    --checkbox-border-color-hover: #1A702B;
    --checkbox-border-color-selected: #1A702B;
    --checkbox-label-text-color: #0F172A;
    --input-background-fill: #F8FAFC;
    --input-background-fill-focus: #FFFFFF;
    --input-background-fill-hover: #F0F9F4;
    --input-border-color: #CBD5E1;
    --input-border-color-focus: #1A702B;
    --input-border-color-hover: #1A702B;
    --input-placeholder-color: #94A3B8;
    --input-text-weight: 500;
    --loader-color: #1A702B;
    --shadow-drop: 0 1px 4px rgba(0,0,0,0.05);
    --shadow-inset: inset 0 1px 2px rgba(0,0,0,0.04);
    --slider-color: #1A702B;
    --table-even-background-fill: #F8FAFC;
    --table-odd-background-fill: #FFFFFF;
    --table-row-focus: #E6F4EA;
    --panel-background-fill: #FFFFFF;
    --panel-border-color: #E2E8F0;
    --button-primary-background-fill: linear-gradient(135deg, #1A702B 0%, #135520 100%);
    --button-primary-background-fill-hover: linear-gradient(135deg, #228B38 0%, #1A702B 100%);
    --button-primary-border-color: #1A702B;
    --button-primary-text-color: #FFFFFF;
    --button-secondary-background-fill: #F0F9F4;
    --button-secondary-border-color: #C3E6CB;
    --button-secondary-text-color: #1A702B;
    --accordion-text-color: #0F172A;
    --stat-background-fill: #F0F9F4;
}

/* ── 2. Global resets ── */
body, .gradio-container {
    background: #F3F7F4 !important;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #0F172A !important;
}

/* ── 3. Layout container ── */
.container { max-width: 1240px; margin: 0 auto; padding-top: 1rem; }

/* ── 4. Header Card ── */
.header-box {
    text-align: center;
    margin-bottom: 1.5rem;
    padding: 2rem 1.5rem 1.6rem;
    background: #FFFFFF;
    border-radius: 18px;
    border: 1px solid #D1EADB;
    box-shadow: 0 4px 24px rgba(26, 112, 43, 0.06);
}
.header-box h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0F172A !important;
    letter-spacing: -0.025em;
    margin-bottom: 0.7rem;
}
.badge-bar { display: flex; justify-content: center; gap: 0.6rem; margin-top: 0.5rem; }

/* ── 5. Section Cards ── */
.section-card {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    border: 1px solid #D1EADB !important;
    padding: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 2px 16px rgba(26, 112, 43, 0.04) !important;
}
.section-card h2 {
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    color: #1A702B !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.6rem !important;
    border-bottom: 2px solid #E6F4EA !important;
}

/* ── 6. Sub-headings ── */
.gradio-container h3, .section-card h3 {
    color: #1E3A2F !important;
    font-weight: 700 !important;
    font-size: 1.0rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 0.75rem !important;
    opacity: 0.7;
}

/* ── 7. Component (block) wrappers — only override background, not borders/radius ── */
/* Don't touch .form/.wrap here — they contain sliders and adding borders creates unwanted boxes */

/* ── 8. Label badges — soft paper-green pill ── */
.gradio-container .block-title,
.gradio-container label > span:first-child {
    background-color: #E6F4EA !important;
    color: #1A702B !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
    padding: 3px 10px !important;
    border-radius: 6px !important;
    border: 1px solid #C3E6CB !important;
    display: inline-block !important;
    margin-bottom: 6px !important;
}

/* ── 9. Text / textarea inputs only (NOT range sliders or number spinners) ── */
.gradio-container input[type="text"],
.gradio-container input[type="email"],
.gradio-container input[type="search"],
.gradio-container textarea,
.gradio-container select {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
    border-color: #1A702B !important;
    box-shadow: 0 0 0 3px rgba(26, 112, 43, 0.12) !important;
    background-color: #FFFFFF !important;
    outline: none !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #94A3B8 !important;
    font-weight: 400 !important;
}

/* ── 9b. Number spinners next to sliders — don't add extra borders ── */
.gradio-container input[type="number"] {
    color: #0F172A !important;
    font-weight: 600 !important;
    min-width: 3.5rem !important;
}

/* ── 9c. Checkbox / toggle active fill ── */
.gradio-container input[type="checkbox"]:checked,
.gradio-container .checkbox input:checked + span,
.gradio-container .toggle-group .selected {
    background-color: #1A702B !important;
    border-color: #1A702B !important;
    accent-color: #1A702B !important;
}
.gradio-container input[type="checkbox"] {
    accent-color: #1A702B !important;
}

/* ── 10. Accordions ── */
.gradio-container details,
.gradio-container details > summary {
    background: #F3F7F4 !important;
    border: 1px solid #D1EADB !important;
    border-radius: 10px !important;
    color: #0F172A !important;
}
.gradio-container details > summary span,
.gradio-container details > summary {
    color: #0F172A !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    background: transparent !important;
}

/* ── 11. Image upload / display panels ── */
.gradio-container [data-testid="image"] {
    background: #F3F7F4 !important;
    border: 1.5px dashed #A7D7B5 !important;
    border-radius: 12px !important;
    max-height: 360px !important;
}
/* UploadText inner wrap (svelte-1vmd51o) — the white rectangle inside the drop zone */
.gradio-container .wrap.svelte-1vmd51o,
.gradio-container [data-testid="image"] .wrap,
.gradio-container [data-testid="image"] > div,
.gradio-container [data-testid="image"] > div > div,
.gradio-container .upload-container.svelte-6uxbr3,
.gradio-container .upload-container.svelte-ey25pz,
.gradio-container .image-container,
.gradio-container .image-frame {
    background: #F3F7F4 !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 12px !important;
}
.gradio-container [data-testid="image"] * {
    color: #475569 !important;
}
.gradio-container [data-testid="image"] svg {
    stroke: #1A702B !important;
    opacity: 0.5;
}

/* ── 11b. Examples strip — precise class kill gap ── */
.gradio-container .examples.svelte-1rn3hyj,
.gradio-container .examples {
    margin-top: 2px !important;
    padding-top: 0 !important;
}
.gradio-container .placeholder.svelte-1rn3hyj,
.gradio-container .placeholder {
    display: none !important;
}

/* ── 12. Primary CTA buttons ── */
.gradio-container button.primary,
.gradio-container button[data-testid="submit-btn"],
button.primary {
    background: linear-gradient(135deg, #1A702B 0%, #135520 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(26, 112, 43, 0.28) !important;
    transition: all 0.2s ease !important;
}
.gradio-container button.primary:hover,
button.primary:hover {
    background: linear-gradient(135deg, #228B38 0%, #1A702B 100%) !important;
    box-shadow: 0 6px 20px rgba(26, 112, 43, 0.38) !important;
    transform: translateY(-1px) !important;
}

/* ── 13. Secondary / icon buttons ── */
.gradio-container button.secondary {
    background: #F0F9F4 !important;
    border: 1px solid #C3E6CB !important;
    color: #1A702B !important;
}

/* ── 14. Scrollbar prevention (only on row container, NOT its children — children clipping cuts min/max numbers) ── */
.gradio-container > .row { overflow-x: hidden !important; }
.dim-slider { min-width: 0 !important; }
/* Let slider min/max number labels breathe */
.gradio-container .range-slider span,
.gradio-container [data-testid="slider"] span {
    overflow: visible !important;
    white-space: nowrap !important;
    min-width: max-content !important;
}

/* ── 15. Image max-height clamp (exclude header logos) ── */
.gradio-container img:not(.logo-img) { object-fit: contain !important; max-height: 360px !important; }
"""


def build_demo():
    # Use Base theme so our CSS variables control everything cleanly
    theme = gr.themes.Base(
        primary_hue="emerald",
        secondary_hue="green",
        neutral_hue="zinc",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="VETO: Image Protection Demo") as demo:
        with gr.Column(elem_classes=["container"]):
            # Header
            gr.HTML(
                f"""
                <div class="header-box">
                    <h1>🛡️ VETO: Towards Protecting Images From Frontier AI Editing</h1>
                    <div class="badge-bar">
                        <a href="#" target="_blank" class="logo-btn" title="Project Page">
                            <img src="{_logo_uri('projectpage.png')}" alt="Project Page" class="logo-img logo-img--lg">
                        </a>
                        <a href="https://github.com/multimodal-ai-lab/veto" target="_blank" class="logo-btn" title="GitHub">
                            <img src="{_logo_uri('github.png')}" alt="GitHub" class="logo-img">
                        </a>
                        <a href="https://arxiv.org" target="_blank" class="logo-btn" title="arXiv Paper">
                            <img src="{_logo_uri('arxiv.png')}" alt="arXiv" class="logo-img">
                        </a>
                        <a href="https://huggingface.co" target="_blank" class="logo-btn" title="HuggingFace">
                            <img src="{_logo_uri('hf.png')}" alt="HuggingFace" class="logo-img">
                        </a>
                    </div>
                </div>
                <style>
                .logo-btn {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    padding: 6px;
                    border-radius: 8px;
                    transition: transform 0.15s ease, box-shadow 0.15s ease;
                    text-decoration: none;
                }}
                .logo-btn:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.12); }}
                .logo-img {{ height: 42px !important; width: auto !important; max-height: 42px !important; object-fit: contain; display: block; }}
                .logo-img--lg {{ height: 46px !important; max-height: 46px !important; }}
                </style>
                """
            )

            # -----------------------------------------------------------------
            # PHASE 1: PROTECTION
            # -----------------------------------------------------------------
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("## Phase 1: Protection")
                with gr.Row():
                    # Left Column: Inputs & Hyperparams (Scale 4:6)
                    with gr.Column(scale=4):
                        gr.Markdown("### Upload & Configure")
                        input_img = gr.Image(
                            label="Original Input Image",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=360,
                        )
                        gr.Examples(
                            examples=[
                                [str(ASSETS_DIR / "examples" / "0.png")],
                                [str(ASSETS_DIR / "examples" / "1.png")],
                                [str(ASSETS_DIR / "examples" / "2.png")],
                            ],
                            inputs=[input_img],
                            label="Example Input Images",
                        )

                        with gr.Accordion("⚙️ Protection Hyperparameters", open=False):
                            epsilon_slider = gr.Slider(
                                minimum=1.0,
                                maximum=16.0,
                                value=4.0,
                                step=0.5,
                                label="Perturbation Budget (ε / 255)",
                            )
                            pgd_steps_slider = gr.Slider(
                                minimum=10,
                                maximum=200,
                                value=100,
                                step=5,
                                label="PGD Optimization Steps",
                            )
                            step_size_slider = gr.Slider(
                                minimum=0.5,
                                maximum=5.0,
                                value=2.0,
                                step=0.5,
                                label="PGD Step Size α (/255)",
                            )
                            with gr.Row(equal_height=True):
                                height_slider = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                    label="Height",
                                    min_width=0,
                                    elem_classes=["dim-slider"],
                                )
                                width_slider = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                    label="Width",
                                    min_width=0,
                                    elem_classes=["dim-slider"],
                                )
                            lock_aspect_checkbox = gr.Checkbox(
                                value=True,
                                label="Keep Aspect Ratio",
                            )

                        protect_btn = gr.Button(
                            "🛡️ Run VETO Protection",
                            variant="primary",
                            size="lg",
                        )

                    # Right Column: Protection Results
                    with gr.Column(scale=6):
                        gr.Markdown("### Protection Results")
                        protected_img_out = gr.Image(
                            label="VETO Protected Image",
                            type="pil",
                            format="png",
                            interactive=False,
                            height=360,
                        )
                        prot_status_out = gr.Textbox(
                            label="Protection Status",
                            interactive=False,
                            lines=3,
                        )

            # Hidden state for lossless in-memory transfer & aspect ratio
            clean_state = gr.State(None)
            protected_state = gr.State(None)
            aspect_ratio_state = gr.State(1.0)

            # -----------------------------------------------------------------
            # PHASE 2: EDITING
            # -----------------------------------------------------------------
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("## Phase 2: Editing")
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("### FLUX.2 Image Editing")
                        edit_prompt_in = gr.Textbox(
                            label="Editing Instruction",
                            placeholder="e.g. Change the color of the hair to vibrant neon blue.",
                            lines=2,
                        )
                        gr.Examples(
                            examples=[
                                ["Make it wear a magician hat."],
                                ["Change style to Pokemon."],
                                ["Replace the car with a motorcycle."],
                                ["Change color to yellow."]
                            ],
                            inputs=[edit_prompt_in],
                            label="Example Editing Instructions",
                        )
                        with gr.Accordion("⚙️ FLUX.2 Editing Settings", open=False):
                            edit_steps_slider = gr.Slider(
                                minimum=10,
                                maximum=50,
                                value=28,
                                step=1,
                                label="FLUX.2 Inference Steps",
                            )
                            guidance_scale_slider = gr.Slider(
                                minimum=1.0,
                                maximum=8.0,
                                value=4.0,
                                step=0.5,
                                label="Guidance Scale",
                            )
                            seed_num = gr.Number(
                                value=42,
                                label="Random Seed",
                                precision=0,
                            )

                        edit_btn = gr.Button(
                            "🧪 Run FLUX.2 Edit Comparison",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=6):
                        gr.Markdown("### Edit Results")
                        with gr.Row():
                            clean_edited_out = gr.Image(
                                label="Edit on Clean Image",
                                type="pil",
                                format="png",
                                interactive=False,
                                height=360,
                            )
                            prot_edited_out = gr.Image(
                                label="Edit on VETO Protected Image",
                                type="pil",
                                format="png",
                                interactive=False,
                                height=360,
                            )

                        edit_status_out = gr.Textbox(
                            label="Editing Comparison Status",
                            interactive=False,
                            lines=2,
                        )

        # Wire Up Event Listeners
        input_img.change(
            fn=update_dims_from_image,
            inputs=[input_img],
            outputs=[height_slider, width_slider, aspect_ratio_state],
        )

        height_slider.release(
            fn=on_height_slider_release,
            inputs=[height_slider, lock_aspect_checkbox, aspect_ratio_state],
            outputs=[width_slider],
        )

        width_slider.release(
            fn=on_width_slider_release,
            inputs=[width_slider, lock_aspect_checkbox, aspect_ratio_state],
            outputs=[height_slider],
        )

        protect_btn.click(
            fn=run_protection_pipeline,
            inputs=[
                input_img,
                epsilon_slider,
                pgd_steps_slider,
                step_size_slider,
                height_slider,
                width_slider,
                gr.State(10),  # protection inner inference steps
                gr.State(4.0),  # protection guidance scale
            ],
            outputs=[protected_img_out, prot_status_out, protected_state, clean_state],
        )

        edit_btn.click(
            fn=run_editing_pipeline,
            inputs=[
                clean_state,
                protected_state,
                edit_prompt_in,
                edit_steps_slider,
                guidance_scale_slider,
                height_slider,
                width_slider,
                seed_num,
            ],
            outputs=[clean_edited_out, prot_edited_out, edit_status_out],
        )

    return demo


if __name__ == "__main__":
    get_shared_models()
    app = build_demo()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[str(ASSETS_DIR)],
    )
