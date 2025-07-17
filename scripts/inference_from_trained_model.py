from hippie.dataloading import EphysDatasetLabeled
from hippie.model import hippieUnimodalCVAE, hippieUnimodalEmbeddingModelCVAE
from utils import get_embeddings

import torch
import pandas as pd
import argparse
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--z_dim",
    type=int,
    default=10,
    required=False,
    help="Dimensionality of the latent space"
)
parser.add_argument(
    '--dataset',
    type=str,
    default="cellexplorer-celltype",
    help="Dataset to perform inference on"
)
parser.add_argument(
    '--wave-checkpoint',
    type=str,
    required=True,
    help="Path to the waveform model checkpoint"
)
parser.add_argument(
    '--time-checkpoint',
    type=str,
    required=True,
    help="Path to the time model checkpoint"
)
parser.add_argument(
    '--output-dir',
    type=str,
    default="./embeddings_results",
    help="Directory to save embeddings and visualizations"
)

args = parser.parse_args()
accelerator = "gpu" if torch.cuda.is_available() else "cpu"


# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)

# Load dataset
print(f"Loading dataset: {args.dataset}")
wf = pd.read_csv(f"datasets/{args.dataset}/waveforms.csv")
wf = wf.dropna(axis=1)
isi = pd.read_csv(f"datasets/{args.dataset}/isi_dist.csv")
isi = isi.dropna(axis=1)

wf = wf.to_numpy()
isi = isi.to_numpy()

# Load metadata if available
labels = None
label_names = None
if os.path.exists(f"datasets/{args.dataset}/labels.csv"):
    metadata = pd.read_csv(f"datasets/{args.dataset}/labels.csv")
    if 'label' in metadata.columns:
        labels = metadata['label'].astype('category').cat.codes.to_numpy()
        label_names = metadata['label'].unique()
        print(f"Found {len(label_names)} unique labels: {label_names}")

# If no labels in metadata, create dummy labels (for dataset embedding)
if labels is None:
    labels = np.zeros(wf.shape[0])
    label_names = ["unknown"]
    print("No labels found, using dummy labels")

# Create datasets
dataset_wave = EphysDatasetLabeled(wf, isi, labels, mode="wave", normalize=False)
dataset_time = EphysDatasetLabeled(wf, isi, labels, mode="time", normalize=False)

# Create data loaders
data_loader_wave = torch.utils.data.DataLoader(
    dataset_wave, batch_size=128, shuffle=False
)
data_loader_time = torch.utils.data.DataLoader(
    dataset_time, batch_size=128, shuffle=False
)

# Load model weights first to inspect architecture
try:
    wave_checkpoint = torch.load(args.wave_checkpoint, map_location=torch.device(accelerator))
    time_checkpoint = torch.load(args.time_checkpoint, map_location=torch.device(accelerator))
    print("Checkpoints loaded successfully")
except Exception as e:
    print(f"Error loading checkpoints: {e}")
    exit(1)

# Extract architecture parameters from checkpoint
def extract_model_params_from_checkpoint(checkpoint):
    """Extract model architecture parameters from checkpoint state_dict"""
    state_dict = checkpoint["state_dict"]
    
    # Extract dims from checkpoint
    z_dim = state_dict["model.z_mean.weight"].shape[0]
    class_hidden_dim = state_dict["model.source_embedding.weight"].shape[1]
    num_classes = state_dict["model.class_embedding.weight"].shape[0]
    num_sources = state_dict["model.source_embedding.weight"].shape[0]
    
    # Extract output_size from decoder - use linear_out instead of linear
    if "model.decoder.linear_out.weight" in state_dict:
        output_size = state_dict["model.decoder.linear_out.weight"].shape[0]
    elif "model.decoder.linear.weight" in state_dict:
        output_size = state_dict["model.decoder.linear.weight"].shape[0]
    else:
        output_size = 50
        print("Warning: decoder linear layer not found in checkpoint, using default 50")
    
    return z_dim, class_hidden_dim, output_size, num_classes, num_sources

# Extract parameters for both models
wave_z_dim, wave_class_hidden_dim, wave_output_size, wave_num_classes, wave_num_sources = extract_model_params_from_checkpoint(wave_checkpoint)
time_z_dim, time_class_hidden_dim, time_output_size, time_num_classes, time_num_sources = extract_model_params_from_checkpoint(time_checkpoint)

print(f"Wave model params: z_dim={wave_z_dim}, class_hidden_dim={wave_class_hidden_dim}, output_size={wave_output_size}, num_classes={wave_num_classes}, num_sources={wave_num_sources}")
print(f"Time model params: z_dim={time_z_dim}, class_hidden_dim={time_class_hidden_dim}, output_size={time_output_size}, num_classes={time_num_classes}, num_sources={time_num_sources}")

# Define the number of sources and classes
num_sources = 5  # Adjust based on your pretrained model
num_classes = len(np.unique(labels))

# Create models with correct architecture
print("Loading models from checkpoints...")
wave_model = hippieUnimodalCVAE(
    z_dim=wave_z_dim, 
    output_size=wave_output_size, 
    class_hidden_dim=wave_class_hidden_dim, 
    num_sources=num_sources, 
    num_classes=num_classes
)
time_model = hippieUnimodalCVAE(
    z_dim=time_z_dim, 
    output_size=time_output_size, 
    class_hidden_dim=time_class_hidden_dim, 
    num_sources=num_sources, 
    num_classes=num_classes
)

wave_model = hippieUnimodalEmbeddingModelCVAE(wave_model)
time_model = hippieUnimodalEmbeddingModelCVAE(time_model)

# Load model weights
try:
    # Handle potential class_embedding mismatch
    if "model.class_embedding.weight" in wave_checkpoint["state_dict"]:
        if wave_checkpoint["state_dict"]["model.class_embedding.weight"].size(0) != num_classes:
            print("Warning: Class embedding size mismatch in wave model. Removing from checkpoint.")
            wave_checkpoint["state_dict"].pop("model.class_embedding.weight")
            
    if "model.class_embedding.weight" in time_checkpoint["state_dict"]:
        if time_checkpoint["state_dict"]["model.class_embedding.weight"].size(0) != num_classes:
            print("Warning: Class embedding size mismatch in time model. Removing from checkpoint.")
            time_checkpoint["state_dict"].pop("model.class_embedding.weight")
    
    wave_model.load_state_dict(wave_checkpoint["state_dict"], strict=False)
    time_model.load_state_dict(time_checkpoint["state_dict"], strict=False)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Set models to evaluation mode
wave_model.eval()
time_model.eval()

# Extract embeddings
print("Extracting embeddings...")
with torch.no_grad():
    waveform_embeddings, isi_embeddings, joint_embeddings = get_embeddings(
        data_loader_wave, data_loader_time, wave_model, time_model
    )

# Save embeddings
print("Saving embeddings...")
embedding_data = {
    'waveform': waveform_embeddings,
    'isi': isi_embeddings,
    'joint': joint_embeddings,
    'labels': labels
}

for name, embeddings in zip(['waveform', 'isi', 'joint'], 
                           [waveform_embeddings, isi_embeddings, joint_embeddings]):
    df = pd.DataFrame(embeddings)
    if labels is not None:
        df['label'] = labels
        if label_names is not None:
            df['label_name'] = pd.Categorical([label_names[i.astype(int)] for i in labels])
    
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{name}_embeddings.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {name} embeddings to {output_path}")

# Generate UMAP visualizations
print("Generating UMAP visualizations...")

def create_umap_plot(embeddings, labels, title, output_path):
    """Create a UMAP visualization of the embeddings."""
    reducer = umap.UMAP(random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    if len(np.unique(labels)) > 1:
        scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                             c=labels, cmap='tab10', alpha=0.7, s=10)
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], alpha=0.7, s=10)
    
    plt.title(title)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Generate UMAP plots for each embedding type
for name, embeddings in zip(['waveform', 'isi', 'joint'], 
                           [waveform_embeddings, isi_embeddings, joint_embeddings]):
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{name}_umap.png")
    create_umap_plot(embeddings, labels, f"{args.dataset} {name} embeddings", output_path)
    print(f"Saved {name} UMAP visualization to {output_path}")

# Optional: Generate paired comparisons between modalities
if labels is not None and len(np.unique(labels)) > 1:
    print("Generating comparison plots...")
    
    # Create a figure with three subplots for the three modalities
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (name, embeddings) in enumerate(zip(['waveform', 'isi', 'joint'], 
                                               [waveform_embeddings, isi_embeddings, joint_embeddings])):
        reducer = umap.UMAP(random_state=42)
        umap_embeddings = reducer.fit_transform(embeddings)
        
        scatter = axs[idx].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                         c=labels, cmap='tab10', alpha=0.7, s=10)
        axs[idx].set_title(f"{name} embeddings")
        axs[idx].set_xlabel('UMAP 1')
        axs[idx].set_ylabel('UMAP 2')
    
    # Add a colorbar
    cbar = fig.colorbar(scatter, ax=axs, label='Label')
    
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, f"{args.dataset}_comparison_umap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison visualization to {output_path}")

print("Inference completed successfully!")
