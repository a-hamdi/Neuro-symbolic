# Neuro-Symbolic AI Model with Triton Optimization

This project implements a Neuro-Symbolic AI system that combines neural perception (using Vision-Language Models) with symbolic reasoning (using a rule-based logic engine), optimized with Triton kernels for efficient computation.

## Project Structure

```
neuro_symbolic/
├── neural_perception/      # Neural perception module (VLM implementation)
├── symbolic_reasoning/     # Symbolic reasoning engine and logic rules
├── triton_kernels/         # Triton-optimized kernels for efficient computation
├── fusion/                 # Fusion module to combine neural and symbolic systems
├── utils/                  # Utility functions and helper modules
├── data/                   # Example data and sample images
├── models/                 # Model weights and configuration files
├── main.py                 # Main entry point
├── test.py                 # Test script for evaluation
└── requirements.txt        # Project dependencies
```

## Components

### 1. Neural Perception Module

- Vision-Language Model (VLM) based on CLIP/LLaVA
- Extracts embeddings from images and text
- Provides high-dimensional tensor representations

### 2. Symbolic Reasoning Module

- Rule-based inference engine using first-order logic
- Ontology-based decision making
- Knowledge graph for structured reasoning

### 3. Triton-Optimized Kernels

- Efficient matrix operations
- Boolean logic operations
- Optimized neural-symbolic fusion

### 4. Fusion Module

- Combines neural embeddings with symbolic logic
- Makes final decisions based on both systems
- Provides explainable AI outputs

## Setup and Installation

1. Clone the repository
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the test script:
   ```
   python test.py
   ```

## Example Usage

```python
from neuro_symbolic import NeuroSymbolicSystem

# Initialize the system
system = NeuroSymbolicSystem()

# Process an image and text
result = system.process(
    image_path="data/cat.jpg",
    text="A cute animal"
)

# Print the results
print(result)
```

## Concept Embeddings

The system uses concept embeddings to match visual content with semantic concepts. We've implemented two approaches:

1. **Real Concept Embeddings** (Default): The system uses pre-computed embeddings generated from example images and text prompts.
2. **Random Embeddings** (Fallback): If real embeddings aren't available, the system falls back to random embeddings.

### Generating Concept Embeddings

To generate real concept embeddings for improved detection:

```
python generate_embeddings.py
```

This script:
- Processes example images for defined concepts
- Extracts embeddings using the neural perception model
- Combines image and text embeddings for each concept
- Saves the embeddings to `data/concept_embeddings.json`

You can add your own concepts by modifying the `concepts` dictionary in `generate_embeddings.py`.

### Configuration

The system's concept detection can be configured by adjusting the similarity threshold:

```python
# Initialize with a custom threshold (default is 0.7)
system = NeuroSymbolicSystem(threshold=0.2)
```

Lower threshold values (e.g., 0.2) make the system more sensitive to detecting concepts, while higher values (e.g., 0.8) require stronger similarity for concept detection.

### Detection Improvement

Using real concept embeddings significantly improves the system's ability to detect concepts in images:

- **With Random Embeddings**: The system rarely detects concepts due to random similarity scores.
- **With Real Embeddings**: The system can accurately detect concepts like "cat", "dog", etc., and make appropriate symbolic inferences.

Example detection output:
```json
{
  "detected_concepts": ["cat", "pet"],
  "symbolic_inference": ["The detected object is a pet", "This image contains a pet."],
  "final_decision": "A pet is detected in the image."
}
```

## Performance

The Triton-optimized kernels provide significant speedups compared to naive PyTorch implementations, especially for the fusion operations between neural and symbolic systems.

## Extending the System

New symbolic rules can be added to `symbolic_reasoning/rules.py` to extend the system's reasoning capabilities. 