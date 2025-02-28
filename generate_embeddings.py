import torch
import os
import json
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_perception.model import NeuralPerceptionModel
from utils.helpers import save_json

def generate_concept_embeddings():
    """
    Generate concept embeddings using example images and save them to a file.
    """
    # Initialize the neural perception model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neural_model = NeuralPerceptionModel(device=device)
    print(f"Neural model initialized on {device}")
    
    # Define concepts and corresponding example images/text
    concepts = {
        "cat": {
            "images": ["data/cat.jpg"],
            "text": "A cat"
        },
        # We'll use text prompts for other concepts since we don't have images
        "dog": {
            "text": "A dog"
        },
        "bird": {
            "text": "A bird"
        },
        "car": {
            "text": "A car"
        },
        "tree": {
            "text": "A tree"
        },
        "building": {
            "text": "A building"
        },
        "person": {
            "text": "A person"
        },
        "pet": {
            "text": "A pet animal"
        }
    }
    
    # Dictionary to store all embeddings
    embeddings_dict = {}
    
    # Process each concept
    for concept, sources in concepts.items():
        print(f"Processing concept: {concept}")
        embeddings = []
        
        # Process images if available
        if "images" in sources:
            for img_path in sources["images"]:
                if os.path.exists(img_path):
                    print(f"  Processing image: {img_path}")
                    result = neural_model.extract_embeddings(image_path=img_path)
                    if "image_embedding" in result:
                        embeddings.append(result["image_embedding"])
        
        # Process text if available
        if "text" in sources:
            print(f"  Processing text: {sources['text']}")
            result = neural_model.extract_embeddings(text=sources["text"])
            if "text_embedding" in result:
                embeddings.append(result["text_embedding"])
        
        # Compute average embedding if we have any
        if embeddings:
            # Convert list of embeddings to tensor
            embeddings_tensor = torch.cat([emb.reshape(1, -1) for emb in embeddings], dim=0)
            # Compute mean embedding
            mean_embedding = torch.mean(embeddings_tensor, dim=0)
            # Normalize
            normalized_embedding = mean_embedding / mean_embedding.norm()
            # Store
            embeddings_dict[concept] = normalized_embedding.tolist()
    
    # Save embeddings to file
    output_path = "data/concept_embeddings.json"
    save_json(embeddings_dict, output_path)
    print(f"Concept embeddings saved to {output_path}")

if __name__ == "__main__":
    generate_concept_embeddings() 