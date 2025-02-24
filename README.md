# HIPPIE
High-dimensional Interpretation for Physiological Patterns in Intercellular Electrophysiology (HIPPIE), a cVAE framework designed for multimodal neuron classification and clustering by integrating extracellular action potential waveforms with spike-timing derived measurements.


To run HIPPIE

git clone https://github.com/braingeneers/Hippie

conda create --name hippie python=3.10

conda activate hippie

pip install -r requirements.txt

Starting from the main folder run with the following parameters:

- z_dim (Latent space dimensionality) = 10
- weight decay = 
- Learning rate = 
- Beta = 
- Dataset (Dataset folder) = cellexplorer-area

python /scripts/train_model.py --z_dim ${HIDDENSIZE} --weight-decay ${WEIGHT_DECAY} --learning-rate ${LEARNING_RATE} --beta=${BETA} --dataset ${DATASET}