# Spectral Graph Convolutional Network Amyloid-Beta Classifier

The code in this repository implements a spectral graph convolutional network to 
train on structural magnetic resonance imaging data to predict amyloid-beta positivity, a hallmark pathology in Alzheimer's disease.

Additions to the SCGN: 
1. Models to accept 3D tensors as input or 2D tensor input with part of its data appended to the fully connected layer

2. Early Stopping

3. Sensitiviy, specificity, postive predictive value, and negative predictive value

4. FC layer tensor embedding for TensorBoard visualization

Other stuff:
1. ADNI data processing and aggregation routines

2. Functions to shuffle and generate training, test, and validation sets

3. Script for running models for training and evaluation
```
   python mci.py --help
   evaluate.py
```

## Input Data

Clinical imaging:
Structural T1-weighted MR Images. Parcellated into 86 regions to obtain volume data.
AV45-PET exam results served as labels for training the model. 

Clinical and demographic data also included as covariates.


## Model Architectures

[schematic here]

### Type 1
Input: 86 volume regions

### Type 2
Input: 86 volume regions at the begining; covariates appended to the FC layer

### Type 3
Input: 86 volume regions and covariates fed at the beginning
```
   python mci.py -m3
```


## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/cysmnl/amyloid_graph
   cd amyloid_graph
   ```

2. Install the dependencies. The code should run with TensorFlow 1.0 and newer.
   ```sh
   pip install -r requirements.txt
   ```