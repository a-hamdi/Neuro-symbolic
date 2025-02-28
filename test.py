import argparse
import os
import json
import time
import torch
from PIL import Image
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np

# Import the main system
from main import NeuroSymbolicSystem

# Import specific components for individual testing
from neural_perception.model import NeuralPerceptionModel
from symbolic_reasoning.logic_engine import SymbolicReasoningEngine
from fusion.fusion_module import NeuroSymbolicFusion
from triton_kernels.matrix_ops import triton_matmul, triton_logical_and, triton_cosine_similarity


def test_neural_perception(image_path: str, text: str) -> Dict[str, Any]:
    """
    Test the neural perception module.
    
    Args:
        image_path: Path to the image file
        text: Text to process
        
    Returns:
        Dictionary containing embeddings
    """
    print("Testing neural perception module...")
    
    # Initialize the model
    model = NeuralPerceptionModel()
    
    # Start timer
    start_time = time.time()
    
    # Extract embeddings
    embeddings = model.extract_embeddings(
        image_path=image_path,
        text=text
    )
    
    # End timer
    end_time = time.time()
    
    # Print results
    print(f"Embedding extraction took {end_time - start_time:.4f} seconds")
    print(f"Image embedding shape: {embeddings.get('image_embedding', torch.tensor([])).shape}")
    print(f"Text embedding shape: {embeddings.get('text_embedding', torch.tensor([])).shape}")
    
    if 'similarity' in embeddings:
        print(f"Image-text similarity: {embeddings['similarity'].item():.4f}")
    
    return embeddings


def test_symbolic_reasoning() -> None:
    """Test the symbolic reasoning module."""
    print("\nTesting symbolic reasoning module...")
    
    # Initialize the engine
    engine = SymbolicReasoningEngine()
    
    # Define some test facts
    test_facts = [
        {
            "detected_object": {
                "type": "cat",
                "properties": ["fluffy", "alive"]
            }
        },
        {
            "detected_object": {
                "type": "dog",
                "properties": ["playful", "alive"]
            }
        },
        {
            "detected_object": {
                "type": "car",
                "properties": ["red", "fast"]
            }
        },
        {
            "detected_object": {
                "properties": ["alive", "owned_by_humans"]
            }
        }
    ]
    
    # Test each fact
    for i, facts in enumerate(test_facts):
        print(f"\nTest case {i+1}:")
        print(f"Initial facts: {facts}")
        
        # Apply reasoning
        inferred_facts = engine.reason(facts)
        
        # Get derived facts
        derived_facts = engine.get_derived_facts(inferred_facts)
        
        print(f"Derived facts: {derived_facts}")


def test_triton_kernels() -> None:
    """Test the Triton-optimized kernels."""
    print("\nTesting Triton-optimized kernels...")
    
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Triton kernel tests.")
        return
    
    # Set device
    device = torch.device('cuda')
    
    # Test matrix multiplication
    print("\nTesting matrix multiplication...")
    M, K, N = 1024, 1024, 1024
    
    # Create random matrices
    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)
    
    # Time PyTorch multiplication
    torch.cuda.synchronize()
    start = time.time()
    torch_result = a @ b
    torch.cuda.synchronize()
    torch_time = time.time() - start
    
    # Time Triton multiplication
    torch.cuda.synchronize()
    start = time.time()
    triton_result = triton_matmul(a, b)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    # Verify correctness
    is_close = torch.allclose(torch_result, triton_result, rtol=1e-2, atol=1e-2)
    
    print(f"Matrix size: {M}x{K} @ {K}x{N}")
    print(f"PyTorch time: {torch_time:.4f}s")
    print(f"Triton time: {triton_time:.4f}s")
    print(f"Speedup: {torch_time / triton_time:.2f}x")
    print(f"Results match: {is_close}")
    
    # Test logical AND
    print("\nTesting logical AND...")
    a_bin = torch.randint(0, 2, (1024, 1024), device=device, dtype=torch.float32)
    b_bin = torch.randint(0, 2, (1024, 1024), device=device, dtype=torch.float32)
    
    # Time PyTorch operation
    torch.cuda.synchronize()
    start = time.time()
    torch_and = (a_bin > 0) & (b_bin > 0)
    torch.cuda.synchronize()
    torch_time = time.time() - start
    
    # Time Triton operation
    torch.cuda.synchronize()
    start = time.time()
    triton_and = triton_logical_and(a_bin, b_bin) > 0.5
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    # Verify correctness
    is_close = torch.all(torch_and == triton_and)
    
    print(f"Tensor size: 1024x1024")
    print(f"PyTorch time: {torch_time:.4f}s")
    print(f"Triton time: {triton_time:.4f}s")
    print(f"Speedup: {torch_time / triton_time:.2f}x")
    print(f"Results match: {is_close}")


def test_full_system(image_path: str, text: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Test the full neuro-symbolic system.
    
    Args:
        image_path: Path to the image file
        text: Text to process
        output_file: Path to output file
        
    Returns:
        Dictionary containing results
    """
    print("\nTesting full neuro-symbolic system...")
    
    # Initialize the system
    system = NeuroSymbolicSystem()
    
    # Start timer
    start_time = time.time()
    
    # Process the input
    results = system.process(
        image_path=image_path,
        text=text
    )
    
    # End timer
    end_time = time.time()
    
    # Print results
    print(f"Processing took {end_time - start_time:.4f} seconds")
    print(f"Detected concepts: {results.get('detected_concepts', [])}")
    print(f"Symbolic inferences: {results.get('symbolic_inference', [])}")
    print(f"Final decision: {results.get('final_decision', '')}")
    
    # Save results if output file is provided
    if output_file:
        system.save_results(results, output_file)
        print(f"Results saved to {output_file}")
    
    return results


def create_example_data():
    """Create example data if not already present."""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create a dummy image (a colored rectangle)
    if not os.path.exists('data/cat.jpg'):
        # Create a simple image (red rectangle)
        img = Image.new('RGB', (224, 224), color = 'white')
        pixels = img.load()
        
        # Draw something that vaguely resembles a cat (very abstract!)
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                # Draw a cat-like shape
                if (i-112)**2 + (j-112)**2 < 80**2:  # Head
                    pixels[i, j] = (200, 180, 160)
                    
                # Ears
                if ((i-80)**2 + (j-80)**2 < 30**2) or ((i-144)**2 + (j-80)**2 < 30**2):
                    pixels[i, j] = (200, 180, 160)
                    
                # Eyes
                if ((i-100)**2 + (j-100)**2 < 10**2) or ((i-124)**2 + (j-100)**2 < 10**2):
                    pixels[i, j] = (0, 100, 200)
                    
                # Nose
                if (i-112)**2 + (j-120)**2 < 8**2:
                    pixels[i, j] = (255, 100, 100)
        
        # Save the image
        img.save('data/cat.jpg')
        print("Created example image: data/cat.jpg")
    
    # Create a sample rules file if it doesn't exist
    if not os.path.exists('data/rules.json'):
        rules = [
            {
                "name": "cat_is_pet",
                "conditions": [{"subject": "detected_object", "predicate": "is_a", "object": "cat"}],
                "actions": [
                    {"subject": "detected_object", "predicate": "is_a", "object": "pet"},
                    {"subject": "image", "predicate": "contains", "object": "pet"}
                ]
            },
            {
                "name": "dog_is_pet",
                "conditions": [{"subject": "detected_object", "predicate": "is_a", "object": "dog"}],
                "actions": [
                    {"subject": "detected_object", "predicate": "is_a", "object": "pet"},
                    {"subject": "image", "predicate": "contains", "object": "pet"}
                ]
            }
        ]
        
        with open('data/rules.json', 'w') as f:
            json.dump(rules, f, indent=2)
        
        print("Created example rules: data/rules.json")


def main():
    """Run the test script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test Neuro-Symbolic AI System')
    parser.add_argument('--image', type=str, default='data/cat.jpg', help='Path to input image')
    parser.add_argument('--text', type=str, default='A cute animal', help='Input text')
    parser.add_argument('--output', type=str, default='test_output.json', help='Path to output file')
    parser.add_argument('--component', type=str, choices=['neural', 'symbolic', 'triton', 'full', 'all'],
                        default='all', help='Component to test')
    
    args = parser.parse_args()
    
    # Create example data if needed
    create_example_data()
    
    # Test the specified component(s)
    if args.component in ['neural', 'all']:
        test_neural_perception(args.image, args.text)
    
    if args.component in ['symbolic', 'all']:
        test_symbolic_reasoning()
    
    if args.component in ['triton', 'all']:
        test_triton_kernels()
    
    if args.component in ['full', 'all']:
        test_full_system(args.image, args.text, args.output)


if __name__ == "__main__":
    # Add a try-except block to handle errors gracefully
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 