# HIPPIE

High-dimensional Interpretation for Physiological Patterns in Intercellular Electrophysiology (HIPPIE), a cVAE framework designed for multimodal neuron classification and clustering by integrating extracellular action potential waveforms with spike-timing derived measurements.

## To run HIPPIE

```bash
git clone https://github.com/braingeneers/Hippie

conda create --name hippie python=3.10

conda activate hippie

pip install -r requirements.txt
```

Starting from the main folder run with the following parameters:

- z_dim (Latent space dimensionality) = 10
- weight decay = 0.01
- Learning rate = 0.001
- Beta = (0.1 - 0.9) 
- Dataset (Dataset folder) = cellexplorer-celltype

```bash
python /scripts/train_model.py --z_dim 10 --weight-decay 0.01 --learning-rate 0.001 --beta 0.5 --dataset cellexplorer-celltype
```

From trained model
```bash
python inference_from_trained_model.py --dataset cellexplorer-celltype \
                   --wave-checkpoint path/to/wave_model.pt \
                   --time-checkpoint path/to/time_model.pt \
                   --joint-checkpoint path/to/joint_model.pt
                   --output-dir ./embeddings_results
```