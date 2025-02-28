import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from typing import Dict, Any, Union, Tuple, Optional


class NeuralPerceptionModel:
    """
    Neural Perception Model based on CLIP for extracting embeddings from images and text.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        """
        Initialize the neural perception model.
        
        Args:
            model_name: The name or path of the CLIP model to use
            device: The device to run the model on (CPU or CUDA). If None, will use CUDA if available.
        """
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading neural perception model on {self.device}...")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        print(f"Neural perception model loaded successfully.")
    
    def extract_embeddings(self, image_path: Optional[str] = None, 
                          image: Optional[Image.Image] = None,
                          text: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from an image and/or text.
        
        Args:
            image_path: Path to the image file
            image: PIL Image object
            text: Text to process
            
        Returns:
            Dictionary containing image and/or text embeddings
        """
        result = {}
        
        # Process inputs
        if image_path is not None and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        
        # Prepare a batch with the available inputs
        inputs = {}
        
        if image is not None:
            inputs['pixel_values'] = self.processor(images=image, return_tensors="pt")['pixel_values'].to(self.device)
        
        if text is not None:
            inputs['input_ids'] = self.processor(text=text, return_tensors="pt", padding=True)['input_ids'].to(self.device)
            inputs['attention_mask'] = self.processor(text=text, return_tensors="pt", padding=True)['attention_mask'].to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            # Only process what we have
            if 'pixel_values' in inputs:
                image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                result['image_embedding'] = image_features
            
            if 'input_ids' in inputs and 'attention_mask' in inputs:
                text_features = self.model.get_text_features(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                result['text_embedding'] = text_features
                
            # If we have both image and text, compute similarity
            if 'image_embedding' in result and 'text_embedding' in result:
                similarity = torch.matmul(result['image_embedding'], result['text_embedding'].T)
                result['similarity'] = similarity
        
        return result
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Dimension of the embeddings
        """
        return self.model.config.projection_dim

# Example usage
if __name__ == "__main__":
    model = NeuralPerceptionModel()
    embeddings = model.extract_embeddings(
        text="A cute cat sitting on a couch",
        image_path="../../data/example.jpg"
    )
    print(f"Image embedding shape: {embeddings.get('image_embedding', torch.tensor([])).shape}")
    print(f"Text embedding shape: {embeddings.get('text_embedding', torch.tensor([])).shape}")
    if 'similarity' in embeddings:
        print(f"Similarity: {embeddings['similarity'].item():.4f}") 