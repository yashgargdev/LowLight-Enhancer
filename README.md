---
title: NightVision AI Low-Light Enhancer
emoji: 🌙
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
---

# NightVision AI: Low-Light Image Enhancer
A custom PyTorch U-Net architecture trained with L1, SSIM, Total Variation, and Color Constancy losses to illuminate extremely dark photographs while preserving high-frequency architectural details and original chromaticity.

### Features
* Built entirely from scratch using PyTorch.
* Employs spatial skip connections to eliminate deep-CNN blurring.
* Suppresses ISO grain via Total Variation penalties.
* Includes a dynamic Light/Dark mode minimalist UI.
* Built-in webcam support.

### Note on Weights
This Space requires the `lowlight_model.pth` tensor dictionary file to be uploaded alongside `app.py`, `model.py`, and `utils.py` to function natively.
