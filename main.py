import argparse
import json
import os
import torch
from typing import Dict, Any, Optional
from PIL import Image

# Import the modules
from neural_perception.model import NeuralPerceptionModel
from symbolic_reasoning.logic_engine import SymbolicReasoningEngine
from fusion.fusion_module import NeuroSymbolicFusion
from utils.helpers import save_json, load_json, timing_decorator, print_system_info


class NeuroSymbolicSystem:
    """
    Main class for the Neuro-Symbolic AI system.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the Neuro-Symbolic system.
        
        Args:
            config_file: Path to JSON configuration file
        """
        # Load configuration if provided
        self.config = self._load_config(config_file)
        
        # Set device
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize neural perception model
        self.neural_model = NeuralPerceptionModel(
            model_name=self.config.get('neural_model', 'openai/clip-vit-base-patch32'),
            device=self.device
        )
        
        # Initialize symbolic reasoning engine
        self.symbolic_engine = SymbolicReasoningEngine(
            rules_file=self.config.get('rules_file', None)
        )
        
        # Initialize fusion module
        self.fusion = NeuroSymbolicFusion(
            neural_model=self.neural_model,
            symbolic_engine=self.symbolic_engine,
            device=self.device,
            threshold=self.config.get('similarity_threshold', 0.7)
        )
        
        print(f"Neuro-Symbolic system initialized on {self.device}")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            Dictionary containing configuration
        """
        # Default configuration
        default_config = {
            'neural_model': 'openai/clip-vit-base-patch32',
            'similarity_threshold': 0.7,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # If no config file is provided, return default config
        if not config_file:
            return default_config
        
        # Load config from file if it exists
        if os.path.exists(config_file):
            try:
                config = load_json(config_file)
                
                # Merge with default config
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                        
                return config
            except Exception as e:
                print(f"Error loading config file: {e}")
                return default_config
        else:
            print(f"Warning: Config file {config_file} not found. Using default configuration.")
            return default_config
    
    @timing_decorator
    def process(self, image_path: Optional[str] = None, 
               image: Optional[Image.Image] = None,
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
        # Delegate processing to the fusion module
        return self.fusion.process(
            image_path=image_path,
            image=image,
            text=text
        )
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save results to a JSON file.
        
        Args:
            results: Dictionary of results
            output_file: Path to output file
        """
        # Use the save_json utility function
        save_json(results, output_file)
        print(f"Results saved to {output_file}")


def main():
    """Main function for command-line usage."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Neuro-Symbolic AI System')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--text', type=str, help='Input text')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output.json', help='Path to output file')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    
    args = parser.parse_args()
    
    # Print system information if verbose
    if args.verbose:
        print_system_info()
    
    # Initialize the system
    system = NeuroSymbolicSystem(config_file=args.config)
    
    # Process the input
    results = system.process(
        image_path=args.image,
        text=args.text
    )
    
    # Print the results
    print("\nDetected concepts:", results.get('detected_concepts', []))
    print("\nSymbolic inferences:", results.get('symbolic_inference', []))
    print("\nFinal decision:", results.get('final_decision', ''))
    
    # Save the results
    system.save_results(results, args.output)


if __name__ == "__main__":
    # Add a try-except block to handle errors gracefully
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 