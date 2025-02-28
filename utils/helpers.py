import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import time
import os
import json
from PIL import Image


def timing_decorator(func):
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that prints execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to output file
    """
    # Create directory if it doesn't exist and if there's a directory path
    dirname = os.path.dirname(file_path)
    if dirname:  # Only try to create directories if there's actually a directory path
        os.makedirs(dirname, exist_ok=True)
    
    # Convert torch tensors to lists for JSON serialization
    serializable_data = {}
    
    def convert_tensor(item):
        if isinstance(item, torch.Tensor):
            return item.tolist()
        elif isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, dict):
            return {k: convert_tensor(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert_tensor(i) for i in item]
        else:
            return item
    
    # Convert data for JSON serialization
    serializable_data = convert_tensor(data)
    
    # Write to file
    with open(file_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing the loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data


def resize_image(image_path: str, max_size: int = 224) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum size of the longest dimension
        
    Returns:
        Resized PIL Image
    """
    # Load the image
    img = Image.open(image_path)
    
    # Calculate the new size while maintaining aspect ratio
    width, height = img.size
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img


def print_system_info() -> Dict[str, Any]:
    """
    Print system information including PyTorch version, CUDA availability, etc.
    
    Returns:
        Dictionary containing system information
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    
    # Print the information
    print("System Information:")
    print(f"PyTorch version: {info['pytorch_version']}")
    print(f"CUDA available: {info['cuda_available']}")
    print(f"CUDA device count: {info['cuda_device_count']}")
    print(f"CUDA device name: {info['cuda_device_name']}")
    
    return info 