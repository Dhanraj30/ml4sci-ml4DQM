# Vision Transformer for HCAL Data Classification

A Mixture-of-Experts Vision Transformer (MoE-ViT) model to classify HCAL DigiOccupancy data from the CMS detector at the Large Hadron Collider.

## Task Description

The evaluation task requires developing a machine learning model using a Vision Transformer architecture to classify images from two synthetic datasets (`Run355456_Dataset.npy` and `Run357479_Dataset.npy`). These datasets contain DigiOccupancy values (hit multiplicity) for the Hadronic Calorimeter (HCAL) at the CMS detector.

## Dataset Description

- **Data Source**: HCAL sub-detector of the CMS detector at LHC
- **What it represents**: DigiOccupancy (hit multiplicity) values recorded over different lumi-sections
- **Dimensions**: Both datasets are (10000, 64, 72) representing (lumi-sections, ieta, iphi)
- **Data characteristics**: Contains many zero-valued entries throughout ieta and iphi coordinates

## Model Architecture

I implemented a **Mixture-of-Experts Vision Transformer (MoE-ViT)** with the following characteristics:

- **Patch Size**: 8×8 pixels
- **Projection Dimension**: 64
- **Number of Transformer Layers**: 4
- **Number of Attention Heads**: 4
- **Number of Experts**: 4
- **Transformer MLP Units**: [128, 64]
- **Final MLP Head Units**: [256, 128]
- **Dropout Rate**: 0.1

The MoE approach enhances the standard ViT by incorporating multiple specialized "expert" neural networks that are conditionally activated based on the input data. This allows the model to specialize in different aspects of the classification task.

## Data Preprocessing

1. Loading both datasets and assigning appropriate labels (0 for Run355456, 1 for Run357479)
2. Reshaping data to include channel dimension (samples, 64, 72, 1)
3. Normalizing data to range [0,1]
4. Splitting data: 80% training and 20% testing
5. Further splitting training data: 80% training and 20% validation

## Model Evaluation

The model's performance is evaluated using:
- **Accuracy**: Overall classification accuracy
- **ROC Curve**: Receiver Operating Characteristic curve
- **AUC Score**: Area Under the ROC Curve

## How to Run

1. **Setup Environment**:
   ```bash
   # Create and activate virtual environment
   python -m venv ml4sci
   source ml4sci/bin/activate  # Or ml4sci\Scripts\activate on Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Download Dataset**:
   - The notebook includes code to automatically download the datasets from:
     - https://cernbox.cern.ch/s/cDOFb5myDHGqRfc
     - https://cernbox.cern.ch/s/n8NvyK2ldUPUxa9

3. **Run the Notebook**:
   ```bash
   jupyter notebook ML4SCI_MoE_ViT_Classification.ipynb
   ```

## Results

The implemented MoE-ViT model achieved:
- Test Accuracy: ~96%
- AUC Score: ~0.98
- Effective separation between the two classes as shown in the ROC curve

## Dependencies

- Python 3.8+
- TensorFlow 2.8+
- TensorFlow Addons
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Jupyter

## Acknowledgements

The datasets were provided by the ML4SCI collaboration.
