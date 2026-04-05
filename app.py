import os
import torch
import gradio as gr
from PIL import Image

from model import LowLightNet
from utils import load_image, preprocess_image, postprocess_image

# Global model instance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LowLightNet().to(device)
model.eval()

MODEL_PATH = "lowlight_model.pth"

# Attempt to load weights if they exist
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print(f"Warning: Model weights '{MODEL_PATH}' not found. Using untrained model.")

def enhance_image(input_image):
    if input_image is None:
        return None
        
    try:
        # Load and preprocess
        image = load_image(input_image)
        tensor = preprocess_image(image).to(device)
        
        # Inference
        with torch.no_grad():
            output_tensor = model(tensor)
            
        # Postprocess
        output_image = postprocess_image(output_tensor)
        
        return output_image
    except Exception as e:
        print(f"Error during enhancement: {e}")
        return None

# Custom CSS for Modern Minimalist UI (Light & Dark Mode)
custom_css = """
/* Light Mode Variables (Default) */
:root {
    --bg-color: #fcfcfc;
    --text-color: #111111;
    --text-muted: #666666;
    --panel-bg: #ffffff;
    --panel-border: #e5e5e5;
    --panel-hover: #cccccc;
    --btn-primary-bg: #111111;
    --btn-primary-text: #ffffff;
    --btn-primary-hover: #333333;
    --btn-secondary-bg: #f5f5f5;
    --btn-secondary-text: #111111;
    --btn-secondary-border: #cccccc;
    --btn-secondary-hover: #e5e5e5;
}

/* Dark Mode Variables */
.dark {
    --bg-color: #0f0f0f;
    --text-color: #f5f5f5;
    --text-muted: #a3a3a3;
    --panel-bg: #141414;
    --panel-border: #2a2a2a;
    --panel-hover: #555555;
    --btn-primary-bg: #ffffff;
    --btn-primary-text: #000000;
    --btn-primary-hover: #e0e0e0;
    --btn-secondary-bg: #1a1a1a;
    --btn-secondary-text: #ffffff;
    --btn-secondary-border: #333333;
    --btn-secondary-hover: #2a2a2a;
}

body {
    background: var(--bg-color) !important;
    font-family: 'Inter', 'Helvetica Neue', sans-serif !important;
    color: var(--text-color) !important;
    transition: background-color 0.3s, color 0.3s;
}

.gradio-container {
    background: var(--bg-color) !important;
    border: none !important;
}

/* Typography */
.main-title {
    text-align: center;
    color: var(--text-color);
    font-weight: 300;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    letter-spacing: 2px;
    animation: fadeIn 1s ease-out;
}

.sub-title {
    text-align: center;
    color: var(--text-muted);
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
    letter-spacing: 1px;
    animation: fadeIn 1.2s ease-out;
}

/* Clean Panels */
.image-panel {
    border-radius: 4px !important;
    border: 1px solid var(--panel-border) !important;
    background: var(--panel-bg) !important;
    transition: all 0.2s ease-in-out;
}
.image-panel:hover {
    border-color: var(--panel-hover) !important;
}

/* Minimalist Buttons */
.btn-primary {
    background: var(--btn-primary-bg) !important;
    color: var(--btn-primary-text) !important;
    border: none !important;
    font-weight: 500 !important;
    border-radius: 4px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: background-color 0.2s, transform 0.1s;
}

.btn-primary:hover {
    background: var(--btn-primary-hover) !important;
    transform: translateY(-1px);
}

.btn-secondary {
    background: var(--btn-secondary-bg) !important;
    color: var(--btn-secondary-text) !important;
    border: 1px solid var(--btn-secondary-border) !important;
    border-radius: 4px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: background-color 0.2s;
}

.btn-secondary:hover {
    background: var(--btn-secondary-hover) !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slow-glow {
    0% { box-shadow: 0 0 0 0 rgba(128, 128, 128, 0.2); }
    50% { box-shadow: 0 0 10px 2px rgba(128, 128, 128, 0.1); }
    100% { box-shadow: 0 0 0 0 rgba(128, 128, 128, 0.2); }
}

.generating {
    animation: slow-glow 2s infinite !important;
    border-color: var(--text-muted) !important;
}
"""

def flag_image(input_img, output_img):
    import time
    if input_img is None and output_img is None:
        return "Nothing to flag."
    os.makedirs("flagged", exist_ok=True)
    timestamp = int(time.time())
    if input_img:
        input_img.save(f"flagged/{timestamp}_input.png")
    if output_img:
        output_img.save(f"flagged/{timestamp}_output.png")
    return "Issue flagged successfully."

with gr.Blocks(css=custom_css, theme=gr.themes.Monochrome()) as demo:
    gr.HTML("<h1 class='main-title'>NightVision AI</h1>")
    gr.HTML("<p class='sub-title'>Minimalist Low-Light Enhancement</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(
                type="pil", 
                label="Original Photo", 
                sources=["upload", "clipboard", "webcam"],
                interactive=True
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear", elem_classes=["btn-secondary"], size="lg")
                enhance_btn = gr.Button("Enhance", elem_classes=["btn-primary"], size="lg")
                
        with gr.Column(scale=1):
            output_img = gr.Image(
                type="pil", 
                label="Enhanced Result", 
                interactive=False
            )
            
            with gr.Row():
                flag_btn = gr.Button("Flag Issue", elem_classes=["btn-secondary"], size="sm")

    # Status text for flagging
    flag_status = gr.HTML("<div style='text-align: right; color: #a3a3a3; font-size: 0.9rem'></div>")

    # Event listeners
    enhance_btn.click(
        fn=enhance_image,
        inputs=input_img,
        outputs=output_img,
        api_name="enhance"
    )
    
    clear_btn.click(
        fn=lambda: (None, None, ""),
        inputs=None,
        outputs=[input_img, output_img, flag_status]
    )
    
    flag_btn.click(
        fn=flag_image,
        inputs=[input_img, output_img],
        outputs=flag_status
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
