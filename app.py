from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms


IMG_SIZE = 128
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "gen_s2p_epoch_30.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels: int = 3, num_features: int = 64, num_residuals: int = 6) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, 7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.Sequential(
            nn.Conv2d(num_features, num_features * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features * 4),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_features * 4) for _ in range(num_residuals)])
        self.up_blocks = nn.Sequential(
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_features * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 2, num_features, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Conv2d(num_features, img_channels, 7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.res_blocks(x)
        x = self.up_blocks(x)
        return torch.tanh(self.last(x))


def _remove_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


@st.cache_resource
def load_generator(model_path: Path) -> Generator:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = Generator()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = _remove_module_prefix(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


PREPROCESS = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def generate_colored_photo(sketch_image: Image.Image, model: Generator) -> Image.Image:
    input_tensor = PREPROCESS(sketch_image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu()

    output_tensor = (output_tensor * 0.5) + 0.5
    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
    output_np = (output_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(output_np)


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Serif:wght@500;600&display=swap');

            :root {
                --ink: #132338;
                --muted-ink: #37506a;
                --brand: #156c94;
                --brand-soft: #d8ecf5;
                --surface: #ffffffcc;
                --line: #16324a1f;
            }

            .stApp {
                background:
                    radial-gradient(circle at 7% 0%, rgba(255, 221, 180, 0.42), transparent 34%),
                    radial-gradient(circle at 96% 12%, rgba(186, 224, 246, 0.52), transparent 40%),
                    linear-gradient(180deg, #f6f8fb 0%, #edf2f6 100%);
            }

            .block-container {
                max-width: 980px;
                padding-top: 1.1rem;
                padding-bottom: 1rem;
            }

            h1, h2, h3 {
                font-family: 'IBM Plex Serif', Georgia, serif;
                letter-spacing: 0.1px;
                color: var(--ink);
            }

            html, body, [class*="css"], [class^="st-"] {
                font-family: 'Manrope', 'Segoe UI', sans-serif;
                color: var(--ink);
            }

            .hero-card {
                border: 1px solid var(--line);
                border-radius: 20px;
                padding: 24px;
                margin-bottom: 1rem;
                background:
                    linear-gradient(120deg, rgba(255, 255, 255, 0.94), rgba(255, 255, 255, 0.82)),
                    linear-gradient(45deg, #f8fcff 0%, #f8f7ff 100%);
                box-shadow: 0 14px 36px rgba(17, 42, 66, 0.12);
            }

            .hero-title {
                font-size: clamp(1.9rem, 1.2rem + 1.8vw, 3rem);
                line-height: 1.12;
                margin-bottom: 0.35rem;
            }

            .hero-subtitle {
                color: var(--muted-ink);
                margin-bottom: 0;
            }

            .status-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 10px;
                margin-top: 10px;
            }

            .status-card {
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 10px 12px;
                background: var(--surface);
            }

            .status-label {
                font-size: 0.76rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                color: #44617b;
                margin-bottom: 4px;
            }

            .status-value {
                font-weight: 700;
                color: #0f2740;
                font-size: 0.97rem;
            }

            .panel {
                border: 1px solid var(--line);
                border-radius: 16px;
                background: var(--surface);
                padding: 10px 10px 6px;
                box-shadow: 0 10px 24px rgba(17, 42, 66, 0.08);
            }

            .note-card {
                border: 1px dashed #2e5a7e66;
                border-radius: 12px;
                background: #f8fdff;
                padding: 12px 14px;
                margin-top: 6px;
                color: #25435d;
            }

            .stButton button, .stDownloadButton button {
                border-radius: 11px;
                border: 1px solid #24557b;
                font-weight: 700;
                transition: all 180ms ease;
            }

            .stButton button:hover, .stDownloadButton button:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 16px rgba(23, 72, 105, 0.22);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f8fbff 0%, #f1f7fb 100%);
                border-right: 1px solid var(--line);
            }

            div[data-testid="stImage"] img {
                border-radius: 14px;
                border: 1px solid #17324a2b;
                box-shadow: 0 12px 26px rgba(17, 42, 66, 0.13);
                max-height: 320px;
                object-fit: contain;
                background: #f8fbfd;
            }

            @media (max-width: 900px) {
                .status-grid {
                    grid-template-columns: 1fr;
                }
                .hero-card {
                    padding: 18px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Sketch To Photo Generator",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    inject_custom_styles()

    st.markdown(
        """
        <div class="hero-card">
            <h1 class="hero-title">Sketch to Photo Generator</h1>
            <p class="hero-subtitle">Generate color-filled photos from line sketches using your trained CycleGAN model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Inference Console")
        st.caption("Model and runtime details")
        st.write(f"Device: {DEVICE.type.upper()}")
        st.write(f"Input Resolution: {IMG_SIZE} x {IMG_SIZE}")
        st.write(f"Checkpoint: {MODEL_PATH.name}")
        st.divider()
        st.caption("Best results")
        st.write("Use clean sketches with clear contours and minimal background noise.")

    try:
        model = load_generator(MODEL_PATH)
    except Exception as exc:
        st.error(f"Unable to load model checkpoint. Details: {exc}")
        st.stop()

    uploaded = st.file_uploader(
        "Upload Sketch Image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Use a clear sketch image for best results.",
    )

    st.markdown(
        f"""
        <div class="status-grid">
            <div class="status-card">
                <div class="status-label">Model</div>
                <div class="status-value">CycleGAN ResNet-6</div>
            </div>
            <div class="status-card">
                <div class="status-label">Runtime</div>
                <div class="status-value">{DEVICE.type.upper()}</div>
            </div>
            <div class="status-card">
                <div class="status-label">Output Size</div>
                <div class="status-value">{IMG_SIZE} x {IMG_SIZE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded is None:
        st.markdown(
            """
            <div class="note-card">
                Upload a sketch image to begin. After upload, click <strong>Generate Colored Photo</strong> to run inference.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    input_image = Image.open(uploaded).convert("RGB")

    if "output_image" not in st.session_state:
        st.session_state["output_image"] = None

    run_col, info_col = st.columns([1, 1], gap="small")
    with run_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Input Sketch")
        st.image(input_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with info_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Generated Photo")
        if st.session_state["output_image"] is not None:
            st.image(st.session_state["output_image"], use_container_width=True)
        else:
            st.info("No generated result yet. Click the button below.")
        st.markdown('</div>', unsafe_allow_html=True)

    action_col, download_col = st.columns([1, 1], gap="medium")
    with action_col:
        if st.button("Generate Colored Photo", type="primary", use_container_width=True):
            with st.spinner("Generating output image..."):
                st.session_state["output_image"] = generate_colored_photo(input_image, model)
            st.rerun()

    with download_col:
        if st.session_state["output_image"] is not None:
            st.download_button(
                label="Download Generated Photo (PNG)",
                data=pil_to_png_bytes(st.session_state["output_image"]),
                file_name="generated_photo.png",
                mime="image/png",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
