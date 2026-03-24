import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def load_image(image_path_or_file):
    """
    Loads an image from path or file object and converts it to PIL Image in RGB format
    """
    if isinstance(image_path_or_file, str):
        image = Image.open(image_path_or_file).convert("RGB")
    elif isinstance(image_path_or_file, np.ndarray):
        if image_path_or_file.shape[-1] == 3:
            image = Image.fromarray(image_path_or_file)
        else:
            image = Image.fromarray(image_path_or_file).convert("RGB")
    else:
        # If already PIL
        image = image_path_or_file.convert("RGB")
    return image

def preprocess_image(image, size=None):
    """
    Preprocess PIL image to tensor, resizing and normalizing.
    If size is None, no resizing is applied.
    """
    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    
    transform = transforms.Compose(transform_list)
    tensor = transform(image).unsqueeze(0) # Add batch dimension
    return tensor

def postprocess_image(tensor):
    """
    Convert PyTorch tensor back to a PIL image
    """
    tensor = tensor.squeeze(0).detach().cpu()
    # Clamp to [0, 1] range just in case
    tensor = torch.clamp(tensor, 0.0, 1.0)
    transform = transforms.ToPILImage()
    image = transform(tensor)
    return image

def save_output_image(image, save_path):
    """
    Saves a PIL Image to the specified path
    """
    image.save(save_path)
