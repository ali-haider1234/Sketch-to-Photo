# Sketch to Photo Generator (CycleGAN + Streamlit)

A deep learning web app that converts hand-drawn sketches into colorized photo-like outputs using a trained CycleGAN generator.

Built with Streamlit for an interactive UI and PyTorch for inference.

---

## Features

- Upload a sketch image (`png`, `jpg`, `jpeg`, `webp`)
- Generate a colorized output image with one click
- Download the generated image as PNG
- Fast local inference with automatic CPU/GPU selection
- Clean dark-themed UI for a better visual experience

---

## Tech Stack

- Python
- Streamlit
- PyTorch
- Torchvision
- Pillow
- NumPy

---

## Project Structure

```text
assignment3.2/
├── app.py
├── requirements.txt
├── gen_s2p_epoch_30.pth
├── gen_p2s_epoch_30.pth
├── DEPLOYMENT.md
└── assignment-3-2 (1).ipynb
```

---

## Run Locally

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd assignment3.2
```

### 2) Create and activate a virtual environment (recommended)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Start the app

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## How to Use

1. Upload a sketch image.
2. Click **Generate Colored Photo**.
3. Preview the output.
4. Download the generated photo as PNG.

---

## Deployment (Streamlit Community Cloud)

1. Push this project to GitHub.
2. Go to https://share.streamlit.io and sign in.
3. Click **New app**.
4. Select your repository and branch.
5. Set **Main file path** to `app.py`.
6. Click **Deploy**.

Detailed steps are also available in `DEPLOYMENT.md`.

---

## Important Notes

- Keep `gen_s2p_epoch_30.pth` in the same directory as `app.py`.
- If model files are too large for normal Git pushes, use Git LFS or external model hosting.
- For best results, upload clean sketches with clear outlines and minimal noise.

---

## Future Improvements

- Add image size/aspect-ratio validation before inference
- Add optional side-by-side before/after export
- Add model selection toggle (Sketch->Photo / Photo->Sketch)
- Add request logging and monitoring for production

---

## License

This project is for educational use (assignment work).
