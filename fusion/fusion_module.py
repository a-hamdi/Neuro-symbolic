import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
import sys
import os
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_perception.model import NeuralPerceptionModel
from symbolic_reasoning.logic_engine import SymbolicReasoningEngine
from triton_kernels.matrix_ops import triton_matmul, triton_cosine_similarity
from utils.helpers import load_json


class NeuroSymbolicFusion:
    """
    Fusion module that combines neural perception with symbolic reasoning.
    """
    
    def __init__(
        self,
        neural_model: Optional[NeuralPerceptionModel] = None,
        symbolic_engine: Optional[SymbolicReasoningEngine] = None,
        device: Optional[str] = None,
        threshold: float = 0.7,
    ):
        """
        Initialize the fusion module.
        
        Args:
            neural_model: Neural perception model
            symbolic_engine: Symbolic reasoning engine
            device: Device to run on
            threshold: Threshold for similarity matching
        """
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize neural perception model if not provided
        if neural_model is None:
            self.neural_model = NeuralPerceptionModel(device=self.device)
        else:
            self.neural_model = neural_model
            
        # Initialize symbolic reasoning engine if not provided
        if symbolic_engine is None:
            self.symbolic_engine = SymbolicReasoningEngine()
        else:
            self.symbolic_engine = symbolic_engine
            
        # Set threshold for similarity matching
        self.threshold = threshold
        
        # Load concept embeddings (pre-computed or generated)
        self.concept_names, self.concept_embeddings = self._load_concept_embeddings()
        
    def _load_concept_embeddings(self) -> Tuple[List[str], torch.Tensor]:
        """
        Load pre-computed embeddings for known concepts.
        If embeddings file doesn't exist, fall back to random embeddings.
        
        Returns:
            Tuple of (concept_names, embeddings tensor)
        """
        embedding_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "data/concept_embeddings.json")
        
        # Try to load pre-computed embeddings
        if os.path.exists(embedding_file):
            try:
                print(f"Loading concept embeddings from {embedding_file}")
                embeddings_dict = load_json(embedding_file)
                
                # Get concept names and embeddings
                concept_names = list(embeddings_dict.keys())
                
                # Convert embeddings to tensor
                embedding_lists = [embeddings_dict[name] for name in concept_names]
                embedding_dim = len(embedding_lists[0])
                embeddings = torch.zeros(len(concept_names), embedding_dim, device=self.device)
                
                for i, emb_list in enumerate(embedding_lists):
                    embeddings[i] = torch.tensor(emb_list, device=self.device)
                
                print(f"Loaded embeddings for {len(concept_names)} concepts")
                return concept_names, embeddings
            except Exception as e:
                print(f"Error loading concept embeddings: {e}")
        
        # Fall back to random embeddings if loading fails
        print("Using random concept embeddings (fallback)")
        concept_names = ["cat", "dog", "bird", "car", "tree", "building", "person", "pet"]
        embedding_dim = self.neural_model.get_embedding_dimension()
        embeddings = torch.randn(len(concept_names), embedding_dim, device=self.device)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        
        return concept_names, embeddings
        
    def process(self, image_path: Optional[str] = None, 
               image: Optional[Any] = None,
               text: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an image and/or text through the neuro-symbolic system.
        
        Args:
            image_path: Path to the image file
            image: PIL Image object
            text: Text to process
            
        Returns:
            Dictionary containing neural embeddings, symbolic inferences, and final decision
        """
        # 1. Extract embeddings using the neural model
        embeddings = self.neural_model.extract_embeddings(
            image_path=image_path,
            image=image,
            text=text
        )
        
        # 2. Match embeddings to concepts using Triton-optimized cosine similarity
        detected_concepts = self._match_embeddings_to_concepts(embeddings)
        
        # 3. Generate initial facts from detected concepts
        facts = self._generate_facts_from_concepts(detected_concepts)
        
        # 4. Apply symbolic reasoning
        inferred_facts = self.symbolic_engine.reason(facts)
        
        # 5. Update knowledge graph
        self.symbolic_engine.update_knowledge_graph(inferred_facts)
        
        # 6. Get derived facts as human-readable strings
        derived_facts = self.symbolic_engine.get_derived_facts(inferred_facts)
        
        # 7. Generate final decision
        final_decision = self._generate_final_decision(inferred_facts, embeddings)
        
        # Return the combined result
        return {
            "neural_embedding": embeddings,
            "detected_concepts": detected_concepts,
            "symbolic_inference": derived_facts,
            "final_decision": final_decision
        }
    
    def _match_embeddings_to_concepts(self, embeddings: Dict[str, torch.Tensor]) -> List[str]:
        """
        Match embeddings to known concepts using Triton-optimized cosine similarity.
        
        Args:
            embeddings: Dictionary containing image and/or text embeddings
            
        Returns:
            List of detected concepts
        """
        detected_concepts = []
        
        # Process image embeddings if available
        if 'image_embedding' in embeddings:
            # Reshape for batch processing (1 x embedding_dim)
            image_embedding = embeddings['image_embedding'].reshape(1, -1)
            
            # Repeat for batch processing (num_concepts x embedding_dim)
            batch_embedding = image_embedding.repeat(len(self.concept_names), 1)
            
            # Compute similarities using Triton
            similarities = triton_cosine_similarity(batch_embedding, self.concept_embeddings)
            
            # Find matches above threshold
            matches = similarities > self.threshold
            
            # Get matched concept names
            for i, match in enumerate(matches):
                if match.item():
                    detected_concepts.append(self.concept_names[i])
        
        # If text is provided, we could also match text embeddings to concepts
        # (similar to the image embedding matching)
        
        return detected_concepts
    
    def _generate_facts_from_concepts(self, detected_concepts: List[str]) -> Dict[str, Any]:
        """
        Generate initial facts from detected concepts.
        
        Args:
            detected_concepts: List of detected concepts
            
        Returns:
            Dictionary of initial facts
        """
        facts = {}
        
        # If no concepts are detected, return empty facts
        if not detected_concepts:
            return facts
        
        # Add detected concepts as facts
        for i, concept in enumerate(detected_concepts):
            # Use the first concept as the main detected object
            if i == 0:
                facts["detected_object"] = {
                    "type": concept,
                    "properties": ["detected"]
                }
            else:
                # Add additional detected objects
                facts[f"additional_object_{i}"] = {
                    "type": concept,
                    "properties": ["detected"]
                }
        
        return facts
    
    def _generate_final_decision(self, inferred_facts: Dict[str, Any], 
                                embeddings: Dict[str, torch.Tensor]) -> str:
        """
        Generate a final decision combining neural and symbolic information.
        
        Args:
            inferred_facts: Dictionary of inferred facts
            embeddings: Dictionary of neural embeddings
            
        Returns:
            Final decision as a string
        """
        # Default decision if no specific inference is made
        default_decision = "No specific objects detected in the image."
        
        # Check if we have image information
        if 'image' in inferred_facts and isinstance(inferred_facts['image'], dict):
            if 'contains' in inferred_facts['image']:
                contained_object = inferred_facts['image']['contains']
                return f"A {contained_object} is detected in the image."
        
        # Check if we have detected objects
        if 'detected_object' in inferred_facts and isinstance(inferred_facts['detected_object'], dict):
            if 'type' in inferred_facts['detected_object']:
                obj_type = inferred_facts['detected_object']['type']
                return f"The image contains a {obj_type}."
        
        # If no specific decision can be made, return default
        return default_decision


# Example usage
if __name__ == "__main__":
    # Initialize the fusion module
    fusion = NeuroSymbolicFusion()
    
    # Process an image
    result = fusion.process(
        image_path="../data/cat.jpg",
        text="A cute pet"
    )
    
    # Print the result
    print("Detected concepts:", result["detected_concepts"])
    print("Symbolic inferences:", result["symbolic_inference"])
    print("Final decision:", result["final_decision"]) 