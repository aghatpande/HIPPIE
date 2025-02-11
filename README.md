# HIPPIE
High-dimensional Interpretation for Physiological Patterns in Intercellular Electrophysiology (HIPPIE), a cVAE framework designed for multimodal neuron classification and clustering by integrating extracellular action potential waveforms with spike-timing derived measurements.


To run HIPPIE

git clone ...

conda create ...

conda activate ...

pip install -r ...


Starting from the main folder run with the following parameters

- z_dim = 
- weight decay
- Learning rate
- Beta
- Dataset

python /scripts/train_model.py --z_dim ${HIDDENSIZE} --weight-decay ${WEIGHT_DECAY} --learning-rate ${LEARNING_RATE} --beta=${BETA} --dataset ${DATASET}