# Deploying the Sketch-to-Photo Streamlit App

## 1) Run Locally

1. Open a terminal in this project folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

## 2) Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io and sign in.
3. Click **New app**.
4. Select your repository and branch.
5. Set **Main file path** to `app.py`.
6. Click **Deploy**.

## 3) Important Notes

- Ensure `gen_s2p_epoch_30.pth` is in the same directory as `app.py`.
- If the model file is too large for GitHub, use Git LFS or host the checkpoint externally and update `MODEL_PATH` in `app.py`.
- For best quality, upload clean sketch images with clear outlines.

## 4) Optional Production Improvements

- Add request logging and monitoring.
- Add image size validation before inference.
- Add GPU-enabled hosting for faster generation.
