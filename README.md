# HIPPIE

High-dimensional Interpretation for Physiological Patterns in Intercellular Electrophysiology (HIPPIE), a cVAE framework designed for multimodal neuron classification and clustering by integrating extracellular action potential waveforms with spike-timing derived measurements.

## To install HIPPIE

```bash
# Clone this repository
git clone https://github.com/braingeneers/Hippie

# cd into the repo parent folder
cd Hippie
# Create a conda environment with python 3.10
conda create --name hippie python=3.10
# Activate the environment
conda activate hippie

# Install the repo
pip install .
```
## To train a HIPPIE model. 

Add the data you want to use to train your model as a new folder in the ./datasets folder. Follow the stablished naming convention

Starting from the parent folder run the training script with the following parameters:

- z_dim (Latent space dimensionality) = 10
- weight decay = 0.01
- Learning rate = 0.001
- Beta = (0.1 - 0.9) 
- Dataset (Dataset folder) = cellexplorer-celltype

```bash
python /scripts/train_model.py --z_dim 10 --weight-decay 0.01 --learning-rate 0.001 --beta 0.5 --dataset cellexplorer-celltype
```
## To get the embeddings and perform inference 

From trained models and dataset without labels

```bash
python inference_from_trained_model.py --dataset cellexplorer-celltype \
                   --wave-checkpoint path/to/wave_model.pt \
                   --time-checkpoint path/to/time_model.pt \
                   --joint-checkpoint path/to/joint_model.pt
                   --output-dir ./embeddings_results
```