# Vision Transformer for HCAL Data Classification

A traditional Vision Transformer (ViT) model to classify HCAL DigiOccupancy data from the CMS detector at the Large Hadron Collider.

## Task Description

The evaluation task requires developing a machine learning model using a Vision Transformer architecture to classify images from two synthetic datasets (`Run355456_Dataset.npy` and `Run357479_Dataset.npy`). These datasets contain DigiOccupancy values (hit multiplicity) for the Hadronic Calorimeter (HCAL) at the CMS detector.

## Dataset Description

* **Data Source**: HCAL sub-detector of the CMS detector at LHC
* **What it represents**: DigiOccupancy (hit multiplicity) values recorded over different lumi-sections
* **Dimensions**: Both datasets are (10000, 64, 72) representing (lumi-sections, ieta, iphi)
* **Data characteristics**: Contains many zero-valued entries throughout ieta and iphi coordinates

## Model Architecture

I implemented a **Traditional Vision Transformer (ViT)** with the following characteristics:

* **Patch Size**: 8×8 pixels
* **Embedding Dimension**: 128
* **Number of Transformer Layers**: 4
* **Number of Attention Heads**: 4
* **MLP Dimension**: 256
* **Dropout Rate**: 0.1

The ViT approach treats the image as a sequence of patches and leverages self-attention mechanisms to capture relationships between different regions of the HCAL detector data, enabling effective classification of the two datasets.

## Data Preprocessing

1. Loading both datasets and assigning appropriate labels (0 for Run355456, 1 for Run357479)
2. Log transformation to handle skewed distribution
3. Normalizing data to range [0,1]
4. Adding channel dimension for PyTorch compatibility
5. Splitting data: 70% training, 15% validation, and 15% testing

## Model Evaluation

The model's performance is evaluated using:

* **Accuracy**: Overall classification accuracy
* **ROC Curve**: Receiver Operating Characteristic curve
* **AUC Score**: Area Under the ROC Curve
* **Confusion Matrix**: To analyze classification errors

## How to Run

1. **Setup Environment**:

```
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

2. **Download Dataset**:
   * The notebook includes code to automatically load the datasets from:
      * https://cernbox.cern.ch/s/cDOFb5myDHGqRfc
      * https://cernbox.cern.ch/s/n8NvyK2ldUPUxa9

3. **Run the Notebook**:

```
jupyter notebook vit_dqm.ipynb
```

## Results

The implemented ViT model achieved:
* High classification accuracy
* Strong AUC score
* Effective separation between the two classes as shown in the ROC curve
* Meaningful attention visualization showing which parts of the HCAL data the model focuses on

## Dependencies

* Python 3.8+
* PyTorch
* NumPy
* Matplotlib
* Scikit-learn
* Seaborn
* Jupyter

## Acknowledgements

The datasets were provided by the ML4SCI collaboration.
