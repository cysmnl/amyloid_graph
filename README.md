# Spectral Graph Convolutional Network Amyloid-Beta Classifier

The code in this repository implements a spectral graph convolutional network to 
train on structural magnetic resonance imaging data and predict AB positivity.

Additions include 
3D permutations added to coarsening.py

sensitiviy, specificity calculations



## Clinical Imaging Data

MR Images
PET 

1. a data matrix where each row is a sample and each column is a feature,
2. a target vector,
3. optionally, an adjacency matrix which encodes the structure as a graph.

## Model Architectures

[schematic here]

### Type 1
Input: 86 nodes




## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/cysmnl/amyloid_graph
   cd amyloid_graph
   ```

2. Install the dependencies. The code should run with TensorFlow 1.0 and newer.
   ```sh
   pip install -r requirements.txt  # or make install
   ```

3. Play with the Jupyter notebooks.
   ```sh
   jupyter notebook
   ```
