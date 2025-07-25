from hippie.dataloading import EphysDataset, EphysDatasetLabeled, BalancedBatchSampler
from hippie.model import hippieUnimodalCVAE, hippieUnimodalEmbeddingModelCVAE, MultiModalCVAE, MultiModalCVAETrainModule
from utils import make_confmat, get_embeddings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd
import argparse
import os
import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

def get_embeddings_multimodal(loader, model):
    """Extract embeddings from a multimodal model."""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for sample in loader:
            embedding = model(sample)[0].detach().cpu().numpy()
            # Normalize embeddings
            embedding = (embedding - np.mean(embedding, axis=1, keepdims=True)) / np.std(embedding, axis=1, keepdims=True)
            all_embeddings.extend(embedding)
    
    return np.array(all_embeddings)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Common arguments
    parser.add_argument("--z_dim", type=int, default=5, help="Dimension of latent space")
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1, help="Weight for KL divergence loss")
    parser.add_argument('--dataset', type=str, default="cellexplorer-celltype")
    parser.add_argument('--upload-model', action='store_true')
    parser.add_argument('--wandb-tag', type=str, default="no_curr_sup_pretrain_data")
    parser.add_argument('--project', type=str, default="HIPPIE")
    parser.add_argument('--finetune-without-labels', type=bool, default=True)
    parser.add_argument('--pretrain-max-epochs', type=int, default=1)
    parser.add_argument('--finetune-max-epochs', type=int, default=1)
    parser.add_argument('--supervised-max-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--supervised-batch-size', type=int, default=64)
    parser.add_argument('--early-stopping-patience', type=int, default=30)
    parser.add_argument('--gradient-clip-val', type=float, default=1.0)
    parser.add_argument('--train-val-split', type=float, default=0.8)
    parser.add_argument('--finetune-split', type=float, default=0.1)
    parser.add_argument('--limit-train-batches', type=float, default=None)
    parser.add_argument('--limit-val-batches', type=float, default=None)
    
    # New arguments for multimodal approach
    parser.add_argument('--model-type', type=str, choices=['unimodal', 'multimodal'], default='unimodal',
                       help='Whether to use separate models for each modality or a joint model')
    parser.add_argument('--mod1-weight', type=float, default=1.0,
                        help='Weight for the waveform modality loss in multimodal model')
    parser.add_argument('--mod2-weight', type=float, default=1.0,
                        help='Weight for the ISI modality loss in multimodal model')
    
    args = parser.parse_args()
    
    # Common setup
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    limit_train_batches = args.limit_train_batches
    limit_val_batches = args.limit_val_batches
    project = args.project
    FINETUNE_WITHOUT_LABELS = args.finetune_without_labels
    
    torch.manual_seed(42)
    
    # Dataset setup similar for both approaches
    dataset_files = {
        "extracellular-mouse-a1": 1,
        "cellexplorer-celltype": 3,
        "cellexplorer-area": 3,
        "juxtacellular-mouse-s1-celltype": 4,
        "juxtacellular-mouse-s1-area": 4,
        "allenscope-neuropixel": 3,
        "neonatal-mouse-brain-slice": 2,
    }
    
    all_dataset_files = dataset_files.copy()
    num_sources = max(all_dataset_files.values()) + 1
    
    # Remove current dataset from pretraining
    if "justacellular" in args.dataset:
        dataset_files.pop("justacellular-mouse-s1-celltype", None)
        dataset_files.pop("juxtacellular-mouse-s1-area", None)
    
    if "cellexplorer" in args.dataset:
        dataset_files.pop("cellexplorer-celltype", None)
        dataset_files.pop("cellexplorer-area", None)
    
    # Load data for pretraining
    all_waveforms = []
    all_isi = []
    labels = []
    
    if args.model_type == 'unimodal':
        #-----------------------------------------------------------------
        # UNIMODAL APPROACH
        #-----------------------------------------------------------------
        datasets_wf = []
        datasets_isi = []
        
        for folder in dataset_files:
            if folder != args.dataset:
                wf = pd.read_csv(f"datasets/{folder}/waveforms.csv")
                isi = pd.read_csv(f"datasets/{folder}/isi_dist.csv")
                
                wf = wf.to_numpy()
                isi = isi.to_numpy()
                source = np.full((wf.shape[0]), dataset_files[folder])
                print(f"Folder {folder} has shapes {wf.shape} and {isi.shape}")
                
                all_waveforms.append(wf)
                all_isi.append(isi)
                labels.append(source)
                
                dataset_wf = EphysDatasetLabeled(wf, isi, source, mode="wave", normalize=False)
                dataset_isi = EphysDatasetLabeled(wf, isi, source, mode="time", normalize=False)
                
                datasets_wf.append(dataset_wf)
                datasets_isi.append(dataset_isi)
        
        labels = np.concatenate(labels, axis=0)
        all_waveform_dataset = torch.utils.data.ConcatDataset(datasets_wf)
        all_isi_dataset = torch.utils.data.ConcatDataset(datasets_isi)
        
        print(f"Total waveforms {all_waveform_dataset.__len__()} and total isi {all_isi_dataset.__len__()}")
        print(f"Num labels {len(labels)}")
        
        # Split datasets
        prop = args.train_val_split
        indices = list(range(len(all_waveform_dataset)))
        train_indices, test_indices = random_split(
            indices, [int(prop * len(indices)), len(indices) - int(prop * len(indices))]
        )
        
        train_dataset_wave = torch.utils.data.Subset(all_waveform_dataset, train_indices)
        test_dataset_wave = torch.utils.data.Subset(all_waveform_dataset, test_indices)
        
        train_dataset_time = torch.utils.data.Subset(all_isi_dataset, train_indices)
        test_dataset_time = torch.utils.data.Subset(all_isi_dataset, test_indices)
        
        train_loader_wave = torch.utils.data.DataLoader(
            train_dataset_wave, batch_size=args.batch_size, shuffle=True
        )
        test_loader_wave = torch.utils.data.DataLoader(
            test_dataset_wave, batch_size=args.batch_size, shuffle=False
        )
        train_loader_time = torch.utils.data.DataLoader(
            train_dataset_time, batch_size=args.batch_size, shuffle=True
        )
        test_loader_time = torch.utils.data.DataLoader(
            test_dataset_time, batch_size=args.batch_size, shuffle=False
        )
        
        # Create and train separate models
        wave_model = hippieUnimodalCVAE(
            z_dim=args.z_dim, output_size=50, class_hidden_dim=5, 
            num_sources=num_sources, num_classes=5
        )
        time_model = hippieUnimodalCVAE(
            z_dim=args.z_dim, output_size=100, class_hidden_dim=5, 
            num_sources=num_sources, num_classes=5
        )
        
        wave_model = hippieUnimodalEmbeddingModelCVAE(
            wave_model, learning_rate=args.learning_rate, weight_decay=args.weight_decay
        )
        time_model = hippieUnimodalEmbeddingModelCVAE(
            time_model, learning_rate=args.learning_rate, weight_decay=args.weight_decay
        )
        
        # Define callbacks
        wave_modelcheckpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        time_modelcheckpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        early_stop_wave = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stopping_patience, mode="min"
        )
        early_stop_time = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stopping_patience, mode="min"
        )
        
        # Train wave model
        wandb_logger1 = pl.loggers.WandbLogger(
            project=project,
            name=f"{args.wandb_tag}{args.dataset}_wave_model_{args.z_dim}",
        )
        trainer_wave = pl.Trainer(
            max_epochs=args.pretrain_max_epochs,
            accelerator=accelerator,
            logger=wandb_logger1,
            callbacks=[wave_modelcheckpoint, early_stop_wave],
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
        )
        trainer_wave.fit(wave_model, train_loader_wave, test_loader_wave)
        
        # Train time model
        wandb_logger2 = pl.loggers.WandbLogger(
            project=project,
            name=f"{args.wandb_tag}{args.dataset}_time_model_{args.z_dim}",
        )
        trainer_time = pl.Trainer(
            max_epochs=args.pretrain_max_epochs,
            accelerator=accelerator,
            logger=wandb_logger2,
            callbacks=[time_modelcheckpoint, early_stop_time],
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            gradient_clip_val=args.gradient_clip_val,
        )
        trainer_time.fit(time_model, train_loader_time, test_loader_time)
        
        # Load best models
        wave_path = wave_modelcheckpoint.best_model_path
        time_path = time_modelcheckpoint.best_model_path
        wave_model.load_state_dict(torch.load(wave_path)["state_dict"])
        time_model.load_state_dict(torch.load(time_path)["state_dict"])
        
        # Get data for fine-tuning
        wf_ft = pd.read_csv(f"datasets/{args.dataset}/waveforms.csv")
        wf_ft = wf_ft.dropna(axis=1)
        isi_ft = pd.read_csv(f"datasets/{args.dataset}/isi_dist.csv")
        isi_ft = isi_ft.dropna(axis=1)
        
        wf_ft = wf_ft.to_numpy()
        isi_ft = isi_ft.to_numpy()
        label_ft = np.full((wf_ft.shape[0]), all_dataset_files[args.dataset])
        
        finetune_dataset_wave = EphysDatasetLabeled(wf_ft, isi_ft, label_ft, mode="wave", normalize=False)
        finetune_dataset_time = EphysDatasetLabeled(wf_ft, isi_ft, label_ft, mode="time", normalize=False)
        
        if FINETUNE_WITHOUT_LABELS:
            prop = args.finetune_split
            indices = list(range(len(finetune_dataset_wave)))
            
            if os.path.exists(f"datasets/{args.dataset}/metadata.csv") and "chip" in args.dataset:
                # Special handling for chip datasets with timestamps
                metadata = pd.read_csv(f"datasets/{args.dataset}/metadata.csv")
                metadata['datetime'] = pd.to_datetime(metadata['datetime']).dt.time
                
                first_times = metadata['datetime'].sort_values().unique()[:10]
                train_indices = metadata[metadata['datetime'].isin(first_times)].index.tolist()
                test_indices = metadata[~metadata['datetime'].isin(first_times)].index.tolist()
            else:
                train_indices, test_indices = random_split(
                    indices, [int(prop * len(indices)), len(indices) - int(prop * len(indices))]
                )
            
            # Fine-tune wave model
            wave_model = hippieUnimodalEmbeddingModelCVAE(
                wave_model.model, learning_rate=(1/10)*args.learning_rate, weight_decay=args.weight_decay
            )
            time_model = hippieUnimodalEmbeddingModelCVAE(
                time_model.model, learning_rate=(1/10)*args.learning_rate, weight_decay=args.weight_decay
            )
            
            train_finetune_dataset = torch.utils.data.Subset(finetune_dataset_wave, train_indices)
            test_finetune_dataset = torch.utils.data.Subset(finetune_dataset_wave, test_indices)
            
            train_finetune_loader_wave = torch.utils.data.DataLoader(
                train_finetune_dataset, batch_size=args.batch_size, shuffle=False
            )
            test_finetune_loader_wave = torch.utils.data.DataLoader(
                test_finetune_dataset, batch_size=args.batch_size, shuffle=False
            )
            
            train_finetune_dataset_time = torch.utils.data.Subset(finetune_dataset_time, train_indices)
            test_finetune_dataset_time = torch.utils.data.Subset(finetune_dataset_time, test_indices)
            train_finetune_loader_time = torch.utils.data.DataLoader(
                train_finetune_dataset_time, batch_size=args.batch_size, shuffle=False
            )
            test_finetune_loader_time = torch.utils.data.DataLoader(
                test_finetune_dataset_time, batch_size=args.batch_size, shuffle=False
            )
            
            # Fine-tune wave model
            trainer_wave = pl.Trainer(
                max_epochs=args.finetune_max_epochs,
                accelerator=accelerator,
                logger=wandb_logger1,
                callbacks=[wave_modelcheckpoint, early_stop_wave],
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
            )
            trainer_wave.fit(wave_model, train_finetune_loader_wave, test_finetune_loader_wave)
            
            # Fine-tune time model
            trainer_time = pl.Trainer(
                max_epochs=args.finetune_max_epochs,
                accelerator=accelerator,
                logger=wandb_logger2,
                callbacks=[time_modelcheckpoint, early_stop_time],
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
                gradient_clip_val=args.gradient_clip_val,
            )
            trainer_time.fit(time_model, train_finetune_loader_time, test_finetune_loader_time)
            
            # Get embeddings from fine-tuned models
            finetune_embeddings_wave, finetune_embeddings_time, finetune_joint_embeddings = get_embeddings(
                train_finetune_loader_wave, train_finetune_loader_time, wave_model, time_model
            )
        else:
            # No finetuning, just get embeddings
            finetune_loader_wave = torch.utils.data.DataLoader(
                finetune_dataset_wave, batch_size=args.batch_size, shuffle=False
            )
            finetune_loader_time = torch.utils.data.DataLoader(
                finetune_dataset_time, batch_size=args.batch_size, shuffle=False
            )
            finetune_embeddings_wave, finetune_embeddings_time, finetune_joint_embeddings = get_embeddings(
                finetune_loader_wave, finetune_loader_time, wave_model, time_model
            )
        
        # Save pretraining embeddings
        wf_dfs = {"embeddings": []}
        isi_dfs = {"embeddings": []}
        joint_dfs = {"embeddings": []}
        
        wf_dfs["embeddings"].extend(finetune_embeddings_wave)
        isi_dfs["embeddings"].extend(finetune_embeddings_time)
        joint_dfs["embeddings"].extend(finetune_joint_embeddings)
        
        wf_df = pd.DataFrame(wf_dfs)
        isi_df = pd.DataFrame(isi_dfs)
        joint_df = pd.DataFrame(joint_dfs)
        
        wf_df.to_csv(f"pretraining_{args.dataset}_waveform_embeddings.csv")
        isi_df.to_csv(f"pretraining_{args.dataset}_isi_embeddings.csv")
        joint_df.to_csv(f"pretraining_{args.dataset}_joint_embeddings.csv")
        
        wandb.log_artifact(f"pretraining_{args.dataset}_waveform_embeddings.csv", name=f"pretraining_{args.dataset}_waveform_embeddings.csv", type=f"pretraining_{args.dataset}_waveform_embeddings.csv")
        wandb.log_artifact(f"pretraining_{args.dataset}_isi_embeddings.csv", name=f"pretraining_{args.dataset}_isi_embeddings.csv", type=f"pretraining_{args.dataset}_isi_embeddings.csv")
        wandb.log_artifact(f"pretraining_{args.dataset}_joint_embeddings.csv", name=f"pretraining_{args.dataset}_joint_embeddings.csv", type=f"pretraining_{args.dataset}_joint_embeddings.csv")
        
        # Load supervised data for training
        dataset = args.dataset
        supervised_wf = pd.read_csv(f"datasets/{dataset}/waveforms.csv").to_numpy()
        supervised_isi = pd.read_csv(f"datasets/{dataset}/isi_dist.csv").to_numpy()
        
        if os.path.exists(f"datasets/{dataset}/labels.csv"):
            labels = pd.read_csv(f"datasets/{dataset}/labels.csv")
            supervised_labels = labels["label"].values
            le = LabelEncoder().fit(supervised_labels)
            supervised_labels = le.transform(supervised_labels)
        else:
            print(f"No labels.csv found for {dataset}")
            supervised_labels = np.zeros(len(supervised_wf))
            le = LabelEncoder().fit(supervised_labels)
        
        # Create train/val split for supervised learning
        indices = list(range(len(supervised_wf)))
        train_size = int(args.train_val_split * len(indices))
        train_indices, val_indices = random_split(indices, [train_size, len(indices) - train_size])
        
        # Extract train/val data
        wf_train = supervised_wf[train_indices]
        wf_val = supervised_wf[val_indices]
        isi_train = supervised_isi[train_indices]
        isi_val = supervised_isi[val_indices]
        label_train = supervised_labels[train_indices]
        label_val = supervised_labels[val_indices]
        
        num_class_labels = len(np.unique(label_train))
        wave_model.model.class_embedding = nn.Embedding(num_class_labels, wave_model.model.class_hidden_dim)
        time_model.model.class_embedding = nn.Embedding(num_class_labels, time_model.model.class_hidden_dim)
        
        label_train_for_embedding = all_dataset_files[dataset] * np.ones_like(label_train)
        label_val_for_embedding = all_dataset_files[dataset] * np.ones_like(label_val)
        
        dataset_train_wave = EphysDatasetLabeled(
            wf_train, isi_train, np.vstack((label_train, label_train_for_embedding)).T, mode="wave", normalize=False
        )
        dataset_train_time = EphysDatasetLabeled(
            wf_train, isi_train, np.vstack((label_train, label_train_for_embedding)).T, mode="time", normalize=False
        )
        
        dataset_val_wave = EphysDatasetLabeled(
            wf_val, isi_val, np.vstack((label_val, label_val_for_embedding)).T, mode="wave", normalize=False
        )
        dataset_val_time = EphysDatasetLabeled(
            wf_val, isi_val, np.vstack((label_val, label_val_for_embedding)).T, mode="time", normalize=False
        )
        
        train_sampler = BalancedBatchSampler(dataset_train_wave, label_train)
        train_loader_wave = torch.utils.data.DataLoader(
            dataset_train_wave, batch_size=args.supervised_batch_size, sampler=train_sampler, num_workers=4
        )
        test_loader_wave = torch.utils.data.DataLoader(
            dataset_val_wave, batch_size=args.supervised_batch_size, shuffle=False, num_workers=4
        )
        train_loader_time = torch.utils.data.DataLoader(
            dataset_train_time, batch_size=args.supervised_batch_size, sampler=train_sampler, num_workers=4
        )
        test_loader_time = torch.utils.data.DataLoader(
            dataset_val_time, batch_size=args.supervised_batch_size, shuffle=False, num_workers=4
        )
        
        # Load models for supervised learning
        wave_model = hippieUnimodalCVAE(z_dim=args.z_dim, output_size=50, class_hidden_dim=5, 
                                        num_sources=num_sources, num_classes=num_class_labels)
        time_model = hippieUnimodalCVAE(z_dim=args.z_dim, output_size=100, class_hidden_dim=5, 
                                        num_sources=num_sources, num_classes=num_class_labels)
        
        wave_seq = torch.load(wave_path)
        wave_seq["state_dict"].pop("model.class_embedding.weight")
        wave_model = hippieUnimodalEmbeddingModelCVAE(wave_model, learning_rate=(1/10)*args.learning_rate, weight_decay=args.weight_decay)
        wave_model.load_state_dict(wave_seq["state_dict"], strict=False)
        
        time_seq = torch.load(time_path)
        time_seq["state_dict"].pop("model.class_embedding.weight")
        time_model = hippieUnimodalEmbeddingModelCVAE(time_model, learning_rate=(1/10)*args.learning_rate, weight_decay=args.weight_decay)
        time_model.load_state_dict(time_seq["state_dict"], strict=False)
        
        # Set up callbacks and loggers for supervised training
        wave_modelcheckpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        time_modelcheckpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        early_stop_wave = pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min")
        early_stop_time = pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min")
        lr_monitor_wave = pl.callbacks.LearningRateMonitor(logging_interval="step")
        lr_monitor_time = pl.callbacks.LearningRateMonitor(logging_interval="step")
        
        wandb_logger1 = pl.loggers.WandbLogger(
            project=project,
            name=f"{args.wandb_tag}{args.dataset}finetune_wave_model_{dataset}",
        )
        trainer_wave = pl.Trainer(
            max_epochs=args.supervised_max_epochs,
            accelerator=accelerator,
            logger=wandb_logger1,
            callbacks=[wave_modelcheckpoint, early_stop_wave, lr_monitor_wave],
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            gradient_clip_val=args.gradient_clip_val,
        )
        trainer_wave.fit(wave_model, train_loader_wave, test_loader_wave)
        
        wandb_logger2 = pl.loggers.WandbLogger(
            project=project,
            name=f"{args.wandb_tag}{args.dataset}finetune_time_mode_{dataset}",
        )
        trainer_time = pl.Trainer(
            max_epochs=args.supervised_max_epochs,
            accelerator=accelerator,
            logger=wandb_logger2,
            callbacks=[time_modelcheckpoint, early_stop_time, lr_monitor_time],
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            gradient_clip_val=args.gradient_clip_val,
        )
        trainer_time.fit(time_model, train_loader_time, test_loader_time)
        
        # Load best models after supervised training
        wave_path = wave_modelcheckpoint.best_model_path
        time_path = time_modelcheckpoint.best_model_path
        wandb.log({"best_epoch_waveform": wave_path, "best_epoch_time": time_path})
        
        wave_seq = torch.load(wave_path)
        wave_model.load_state_dict(wave_seq["state_dict"])
        wave_model.optimizer.load_state_dict(wave_seq["optimizer_states"][0])
        
        time_seq = torch.load(time_path)
        time_model.load_state_dict(time_seq["state_dict"])
        time_model.optimizer.load_state_dict(time_seq["optimizer_states"][0])
        
        time_model.eval()
        wave_model.eval()
        
        # Get embeddings for evaluation
        train_loader_wave = torch.utils.data.DataLoader(dataset_train_wave, batch_size=128)
        train_loader_time = torch.utils.data.DataLoader(dataset_train_time, batch_size=128)
        
        waveform_embeddings_train, isi_dist_embeddings_train, joint_embeddings_train = get_embeddings(
            train_loader_wave, train_loader_time, wave_model, time_model
        )
        
        waveform_embeddings_test, isi_dist_embeddings_test, joint_embeddings_test = get_embeddings(
            test_loader_wave, test_loader_time, wave_model, time_model
        )
        
        # Evaluate using KNN with different neighbor counts
        joint_bal_accuracy = []
        waveform_bal_accuracy = []
        isi_bal_accuracy = []
        neighbor_options = list(range(5, 20))
        
        for neighbor in neighbor_options:
            print("KNN with", neighbor, "neighbors")
            
            # Joint embeddings
            knn = KNeighborsClassifier(n_neighbors=neighbor)
            knn.fit(joint_embeddings_train, label_train)
            pred = knn.predict(joint_embeddings_test)
            joint_bal_accuracy.append(balanced_accuracy_score(label_val, pred))
        
            # Waveform embeddings
            knn = KNeighborsClassifier(n_neighbors=neighbor)
            knn.fit(waveform_embeddings_train, label_train)
            pred = knn.predict(waveform_embeddings_test)
            waveform_bal_accuracy.append(balanced_accuracy_score(label_val, pred))
        
            # ISI embeddings
            knn = KNeighborsClassifier(n_neighbors=neighbor)
            knn.fit(isi_dist_embeddings_train, label_train)
            pred = knn.predict(isi_dist_embeddings_test)
            isi_bal_accuracy.append(balanced_accuracy_score(label_val, pred))
        
        # Get best results and confusion matrices
        label_names = le.classes_
        
        best_neighbors_waveform = neighbor_options[np.argmax(waveform_bal_accuracy)]
        knn = KNeighborsClassifier(n_neighbors=best_neighbors_waveform)
        knn.fit(waveform_embeddings_train, label_train)
        pred_waveform = knn.predict(waveform_embeddings_test)
        confmat_waveform = confusion_matrix(label_val, pred_waveform)
        
        best_neighbors_isi = neighbor_options[np.argmax(isi_bal_accuracy)]
        knn = KNeighborsClassifier(n_neighbors=best_neighbors_isi)
        knn.fit(isi_dist_embeddings_train, label_train)
        pred_isi = knn.predict(isi_dist_embeddings_test)
        confmat_isi = confusion_matrix(label_val, pred_isi)
        
        best_neighbors_joint = neighbor_options[np.argmax(joint_bal_accuracy)]
        knn = KNeighborsClassifier(n_neighbors=best_neighbors_joint)
        knn.fit(joint_embeddings_train, label_train)
        pred_joint = knn.predict(joint_embeddings_test)
        confmat_joint = confusion_matrix(label_val, pred_joint)
        
        # Save results and log to wandb
        wf_dfs = {"pred": le.inverse_transform(pred_waveform.astype(int)), "true": le.inverse_transform(label_val.astype(int))}
        isi_dfs = {"pred": le.inverse_transform(pred_isi.astype(int)), "true": le.inverse_transform(label_val.astype(int))}
        joint_dfs = {"pred": le.inverse_transform(pred_joint.astype(int)), "true": le.inverse_transform(label_val.astype(int))}
        
        wf_df = pd.DataFrame(wf_dfs)
        isi_df = pd.DataFrame(isi_dfs)
        joint_df = pd.DataFrame(joint_dfs)
        
        wf_df.to_csv(f"{dataset}_waveform_knn.csv")
        isi_df.to_csv(f"{dataset}_isi_knn.csv")
        joint_df.to_csv(f"{dataset}_joint_knn.csv")
        
        wandb.log_artifact(f"{dataset}_waveform_knn.csv", name=f"{dataset}_waveform_knn.csv", type=f"{dataset}_waveform_knn.csv")
        wandb.log_artifact(f"{dataset}_isi_knn.csv", name=f"{dataset}_isi_knn.csv", type=f"{dataset}_isi_knn.csv")
        wandb.log_artifact(f"{dataset}_joint_knn.csv", name=f"{dataset}_joint_knn.csv", type=f"{dataset}_joint_knn.csv")
        
        # Save embeddings for all data
        all_wf_dataloader = torch.utils.data.DataLoader(
            EphysDatasetLabeled(supervised_wf, supervised_isi, 
                                np.vstack((supervised_labels, np.ones_like(supervised_labels) * all_dataset_files[dataset])).T, 
                                mode="wave", normalize=False),
            batch_size=128
        )
        
        all_isi_dataloader = torch.utils.data.DataLoader(
            EphysDatasetLabeled(supervised_wf, supervised_isi, 
                                np.vstack((supervised_labels, np.ones_like(supervised_labels) * all_dataset_files[dataset])).T, 
                                mode="time", normalize=False),
            batch_size=128
        )
        
        wf_embeddings, isi_embeddings, joint_embeddings = get_embeddings(
            all_wf_dataloader, all_isi_dataloader, wave_model, time_model
        )
        
        wf_embeddings_df = pd.DataFrame(wf_embeddings)
        isi_embeddings_df = pd.DataFrame(isi_embeddings)
        joint_embeddings_df = pd.DataFrame(joint_embeddings)
        
        wf_embeddings_df["label"] = le.inverse_transform(supervised_labels.astype(int))
        isi_embeddings_df["label"] = le.inverse_transform(supervised_labels.astype(int))
        joint_embeddings_df["label"] = le.inverse_transform(supervised_labels.astype(int))
        
        wf_embeddings_df.to_csv(f"{dataset}_waveform_embeddings.csv")
        isi_embeddings_df.to_csv(f"{dataset}_isi_embeddings.csv")
        joint_embeddings_df.to_csv(f"{dataset}_joint_embeddings.csv")
        
        wandb.log_artifact(f"{dataset}_waveform_embeddings.csv", name=f"{dataset}_waveform_embeddings.csv", type=f"{dataset}_waveform_embeddings.csv")
        wandb.log_artifact(f"{dataset}_isi_embeddings.csv", name=f"{dataset}_isi_embeddings.csv", type=f"{dataset}_isi_embeddings.csv")
        wandb.log_artifact(f"{dataset}_joint_embeddings.csv", name=f"{dataset}_joint_embeddings.csv", type=f"{dataset}_joint_embeddings.csv")
        
        # Upload models if requested
        if args.upload_model:
            wandb.log_artifact(wave_path, name=f'wave_model_ft_d{args.dataset}_z{args.z_dim}_lr{args.learning_rate}.pt', type='model')
            wandb.log_artifact(time_path, name=f'time_model_ft_d{args.dataset}_z{args.z_dim}_lr{args.learning_rate}.pt', type='model')
        
        # Log final metrics
        wandb.log({
            "best_balanced_accuracy_waveform": np.max(waveform_bal_accuracy),
            "best_balanced_accuracy_isi": np.max(isi_bal_accuracy),
            "best_balanced_accuracy_joint": np.max(joint_bal_accuracy),
        })
        
        # Make confusion matrix figures
        figure_waveform = make_confmat(confmat_waveform, label_names, best_neighbors_waveform)
        figure_isi = make_confmat(confmat_isi, label_names, best_neighbors_isi)
        figure_joint = make_confmat(confmat_joint, label_names, best_neighbors_joint)
        
        # Log confusion matrices as images
        wandb.log({
            f"{dataset}_confusion_matrix_waveform": wandb.Image(figure_waveform),
            f"{dataset}_confusion_matrix_isi": wandb.Image(figure_isi),
            f"{dataset}_confusion_matrix_joint": wandb.Image(figure_joint),
        })
        
    else:
        #-----------------------------------------------------------------
        # MULTIMODAL APPROACH
        #-----------------------------------------------------------------
        datasets_joint = []
        
        for folder in dataset_files:
            if folder != args.dataset:  # Don't use the dataset we test on for pretraining
                wf = pd.read_csv(f"datasets/{folder}/waveforms.csv")
                isi = pd.read_csv(f"datasets/{folder}/isi_dist.csv")
                
                wf = wf.to_numpy()
                isi = isi.to_numpy()
                source = np.full((wf.shape[0]), dataset_files[folder])
                print(f"Folder {folder} has shapes {wf.shape} and {isi.shape}")
                
                all_waveforms.append(wf)
                all_isi.append(isi)
                labels.append(source)
                
                dataset_joint = EphysDatasetLabeled(wf, isi, source, mode="both", normalize=False)
                datasets_joint.append(dataset_joint)
        
        labels = np.concatenate(labels, axis=0)
        all_joint_dataset = torch.utils.data.ConcatDataset(datasets_joint)
        
        # Split datasets
        prop = args.train_val_split
        indices = list(range(len(all_joint_dataset)))
        train_indices, test_indices = random_split(
            indices, [int(prop * len(indices)), len(indices) - int(prop * len(indices))]
        )
        
        # Create dataloaders
        train_joint_dataset = torch.utils.data.Subset(all_joint_dataset, train_indices)
        test_joint_dataset = torch.utils.data.Subset(all_joint_dataset, test_indices)
        
        train_loader_joint = torch.utils.data.DataLoader(
            train_joint_dataset, batch_size=args.batch_size, shuffle=True
        )
        test_loader_joint = torch.utils.data.DataLoader(
            test_joint_dataset, batch_size=args.batch_size, shuffle=False
        )
        
        # Create multimodal model
        joint_model = MultiModalCVAE(
            z_dim=args.z_dim, 
            output_size_wave=50, 
            output_size_isi=100,
            class_hidden_dim=5, 
            num_sources=num_sources, 
            num_classes=5
        )
        
        joint_model = MultiModalCVAETrainModule(
            joint_model, 
            learning_rate=args.learning_rate, 
            weight_decay=args.weight_decay, 
            beta=args.beta,
            mod1_weight=args.mod1_weight,
            mod2_weight=args.mod2_weight
        )
        
        # Set up callbacks and trainer
        joint_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="val_loss", save_top_k=1, mode="min"
        )
        joint_earlystop = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stopping_patience, mode="min"
        )
        
        wandb_logger1 = pl.loggers.WandbLogger(
            project=project,
            name=f"{args.wandb_tag}{args.dataset}_joint_model_{args.z_dim}",
        )
        
        joint_trainer = pl.Trainer(
            max_epochs=args.pretrain_max_epochs,
            accelerator=accelerator,
            logger=wandb_logger1,
            callbacks=[joint_checkpoint, joint_earlystop],
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            gradient_clip_val=args.gradient_clip_val,
        )
        
        # Train joint model
        joint_trainer.fit(joint_model, train_loader_joint, test_loader_joint)
        joint_path = joint_checkpoint.best_model_path
        joint_model.load_state_dict(torch.load(joint_path)["state_dict"])
        
        # Get data for fine-tuning
        wf_ft = pd.read_csv(f"datasets/{args.dataset}/waveforms.csv")
        wf_ft = wf_ft.dropna(axis=1)
        isi_ft = pd.read_csv(f"datasets/{args.dataset}/isi_dist.csv")
        isi_ft = isi_ft.dropna(axis=1)
        
        wf_ft = wf_ft.to_numpy()
        isi_ft = isi_ft.to_numpy()
        label_ft = np.full((wf_ft.shape[0]), all_dataset_files[args.dataset])
        
        finetune_dataset_joint = EphysDatasetLabeled(wf_ft, isi_ft, label_ft, mode="both", normalize=False)
        
        if FINETUNE_WITHOUT_LABELS:
            prop = args.finetune_split
            indices = list(range(len(finetune_dataset_joint)))
            
            if os.path.exists(f"datasets/{args.dataset}/metadata.csv") and "chip" in args.dataset:
                # Special handling for chip datasets with timestamps
                metadata = pd.read_csv(f"datasets/{args.dataset}/metadata.csv")
                metadata['datetime'] = pd.to_datetime(metadata['datetime']).dt.time
                
                first_times = metadata['datetime'].sort_values().unique()[:10]
                train_indices = metadata[metadata['datetime'].isin(first_times)].index.tolist()
                test_indices = metadata[~metadata['datetime'].isin(first_times)].index.tolist()
            else:
                train_indices, test_indices = random_split(
                    indices, [int(prop * len(indices)), len(indices) - int(prop * len(indices))]
                )
            
            # Create new model instance for fine-tuning with lower learning rate
            joint_model = MultiModalCVAETrainModule(
                joint_model.model, 
                learning_rate=(1/10)*args.learning_rate, 
                weight_decay=args.weight_decay,
                beta=args.beta,
                mod1_weight=args.mod1_weight,
                mod2_weight=args.mod2_weight
            )
            
            # Create dataloaders for fine-tuning
            train_finetune_dataset = torch.utils.data.Subset(finetune_dataset_joint, train_indices)
            test_finetune_dataset = torch.utils.data.Subset(finetune_dataset_joint, test_indices)
            
            train_finetune_loader_joint = torch.utils.data.DataLoader(
                train_finetune_dataset, batch_size=args.batch_size, shuffle=False
            )
            test_finetune_loader_joint = torch.utils.data.DataLoader(
                test_finetune_dataset, batch_size=args.batch_size, shuffle=False
            )
            
            # Fine-tune joint model
            joint_trainer = pl.Trainer(
                max_epochs=args.finetune_max_epochs,
                accelerator=accelerator,
                logger=wandb_logger1,
                callbacks=[joint_checkpoint, joint_earlystop],
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
                gradient_clip_val=args.gradient_clip_val,
            )
            
            joint_trainer.fit(joint_model, train_finetune_loader_joint, test_finetune_loader_joint)
            joint_path = joint_checkpoint.best_model_path
            joint_model.load_state_dict(torch.load(joint_path)["state_dict"])
            
            # Get embeddings using the fine-tuned model
            finetune_embeddings_joint = get_embeddings_multimodal(test_finetune_loader_joint, joint_model)
        else:
            # No fine-tuning, just get embeddings from pretrained model
            finetune_loader_joint = torch.utils.data.DataLoader(
                finetune_dataset_joint, batch_size=args.batch_size, shuffle=False
            )
            finetune_embeddings_joint = get_embeddings_multimodal(finetune_loader_joint, joint_model)
        
        # Save pretraining embeddings
        joint_dfs = {"embeddings": []}
        joint_dfs["embeddings"].extend(finetune_embeddings_joint)
        joint_df = pd.DataFrame(joint_dfs)
        joint_df.to_csv(f"pretraining_{args.dataset}_joint_embeddings.csv")
        wandb.log_artifact(
            f"pretraining_{args.dataset}_joint_embeddings.csv", 
            name=f"pretraining_{args.dataset}_joint_embeddings.csv", 
            type=f"pretraining_{args.dataset}_joint_embeddings.csv"
        )
        
        # Supervised learning with k-fold cross-validation
        dataset = args.dataset
        # Regular supervised learning without k-fold cross-validation
        # Load supervised data
        supervised_wf = pd.read_csv(f"datasets/{dataset}/waveforms.csv").to_numpy()
        supervised_isi = pd.read_csv(f"datasets/{dataset}/isi_dist.csv").to_numpy()
        
        if os.path.exists(f"datasets/{dataset}/labels.csv"):
            labels = pd.read_csv(f"datasets/{dataset}/labels.csv")
            supervised_labels = labels["label"].values
            le = LabelEncoder().fit(supervised_labels)
            supervised_labels = le.transform(supervised_labels)
        else:
            print(f"No labels.csv found for {dataset}")
            supervised_labels = np.zeros(len(supervised_wf))
            le = LabelEncoder().fit(supervised_labels)
        
        # Create train/val split
        indices = list(range(len(supervised_wf)))
        train_size = int(args.train_val_split * len(indices))
        train_indices, val_indices = random_split(indices, [train_size, len(indices) - train_size])
        
        # Extract train/val data
        wf_train = supervised_wf[train_indices]
        wf_val = supervised_wf[val_indices]
        isi_train = supervised_isi[train_indices]
        isi_val = supervised_isi[val_indices]
        label_train = supervised_labels[train_indices]
        label_val = supervised_labels[val_indices]
        
        num_class_labels = len(np.unique(label_train))
        
        # Initialize new model with correct number of classes
        supervised_joint_model = MultiModalCVAE(
            z_dim=args.z_dim, 
            output_size_wave=50, 
            output_size_isi=100, 
            class_hidden_dim=5, 
            num_sources=num_sources, 
            num_classes=num_class_labels
        )
        
        # Load pretrained weights but skip the class embedding layer
        joint_seq = torch.load(joint_path)
        joint_seq["state_dict"].pop("model.class_embedding.weight")
        
        supervised_joint_model = MultiModalCVAETrainModule(
            supervised_joint_model, 
            learning_rate=(1/10)*args.learning_rate, 
            weight_decay=args.weight_decay,
            beta=args.beta,
            mod1_weight=args.mod1_weight,
            mod2_weight=args.mod2_weight
        )
        supervised_joint_model.load_state_dict(joint_seq["state_dict"], strict=False)
        
        # Create source labels for embedding
        label_train_for_embedding = all_dataset_files[dataset] * np.ones_like(label_train)
        label_val_for_embedding = all_dataset_files[dataset] * np.ones_like(label_val)
        
        # Create supervised training datasets
        dataset_train_joint = EphysDatasetLabeled(
            wf_train, isi_train, np.vstack((label_train, label_train_for_embedding)).T, 
            mode="both", normalize=False
        )
        dataset_val_joint = EphysDatasetLabeled(
            wf_val, isi_val, np.vstack((label_val, label_val_for_embedding)).T, 
            mode="both", normalize=False
        )
        
        # Use balanced sampling
        train_sampler = BalancedBatchSampler(dataset_train_joint, label_train)
        train_loader_joint = torch.utils.data.DataLoader(
            dataset_train_joint, batch_size=args.supervised_batch_size, 
            sampler=train_sampler, num_workers=4
        )
        test_loader_joint = torch.utils.data.DataLoader(
            dataset_val_joint, batch_size=args.supervised_batch_size, 
            shuffle=False, num_workers=4
        )
        
        # Set up callbacks for supervised training
        joint_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="val_loss", save_top_k=1, mode="min"
        )
        joint_earlystop = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stopping_patience, mode="min"
        )
        lr_monitor_joint = pl.callbacks.LearningRateMonitor(logging_interval="step")
        
        wandb_logger_supervised = pl.loggers.WandbLogger(
            project=project,
            name=f"{args.wandb_tag}{args.dataset}finetune_joint_model",
        )
        
        # Train supervised model
        joint_trainer = pl.Trainer(
            max_epochs=args.supervised_max_epochs,
            accelerator=accelerator,
            logger=wandb_logger_supervised,
            callbacks=[joint_checkpoint, joint_earlystop, lr_monitor_joint],
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            gradient_clip_val=args.gradient_clip_val,
        )
        
        joint_trainer.fit(supervised_joint_model, train_loader_joint, test_loader_joint)
        
        # Load best model checkpoint
        joint_path = joint_checkpoint.best_model_path
        wandb.log({"best_epoch_joint": joint_path})
        
        joint_seq = torch.load(joint_path)
        supervised_joint_model.load_state_dict(joint_seq["state_dict"])
        supervised_joint_model.eval()
        
        # Get embeddings for evaluation
        train_loader_joint_eval = torch.utils.data.DataLoader(dataset_train_joint, batch_size=128)
        joint_embeddings_train = get_embeddings_multimodal(train_loader_joint_eval, supervised_joint_model)
        joint_embeddings_test = get_embeddings_multimodal(test_loader_joint, supervised_joint_model)
        
        # Evaluate using KNN with different neighbor counts
        joint_bal_accuracy = []
        neighbor_options = list(range(5, 20))
        
        for neighbor in neighbor_options:
            print(f"KNN with {neighbor} neighbors")
            knn = KNeighborsClassifier(n_neighbors=neighbor)
            knn.fit(joint_embeddings_train, label_train)
            pred = knn.predict(joint_embeddings_test)
            joint_bal_accuracy.append(balanced_accuracy_score(label_val, pred))
        
        # Get confusion matrix for best number of neighbors
        label_names = le.classes_
        best_neighbors_joint = neighbor_options[np.argmax(joint_bal_accuracy)]
        
        knn = KNeighborsClassifier(n_neighbors=best_neighbors_joint)
        knn.fit(joint_embeddings_train, label_train)
        pred = knn.predict(joint_embeddings_test)
        
        conf_matrix = confusion_matrix(label_val, pred)
        
        # Save predictions and true labels
        joint_dfs = {"pred": le.inverse_transform(pred.astype(int)), "true": le.inverse_transform(label_val.astype(int))}
        joint_df = pd.DataFrame(joint_dfs)
        joint_df.to_csv(f"{dataset}_joint_knn.csv")
        wandb.log_artifact(f"{dataset}_joint_knn.csv", name=f"{dataset}_joint_knn.csv", type=f"{dataset}_joint_knn.csv")
        
        # Save embeddings for all data
        all_dataloader = torch.utils.data.DataLoader(
            EphysDatasetLabeled(supervised_wf, supervised_isi, 
                                np.vstack((supervised_labels, np.ones_like(supervised_labels) * all_dataset_files[dataset])).T, 
                                mode="both", normalize=False),
            batch_size=128
        )
        
        all_embeddings = get_embeddings_multimodal(all_dataloader, supervised_joint_model)
        all_embeddings_df = pd.DataFrame(all_embeddings)
        all_embeddings_df["label"] = le.inverse_transform(supervised_labels.astype(int))
        all_embeddings_df.to_csv(f"{dataset}_joint_embeddings.csv")
        wandb.log_artifact(f"{dataset}_joint_embeddings.csv", name=f"{dataset}_joint_embeddings.csv", type=f"{dataset}_joint_embeddings.csv")
        
        # Upload model if requested
        if args.upload_model:
            wandb.log_artifact(joint_path, name=f'joint_model_ft_d{args.dataset}_z{args.z_dim}_lr{args.learning_rate}.pt', type='model')
        
        # Log performance metrics
        wandb.log({
            "best_balanced_accuracy_joint": np.max(joint_bal_accuracy),
        })
        
        # Make confusion matrix figure
        figure_joint = make_confmat(conf_matrix, label_names, best_neighbors_joint)
        
        # Log confusion matrix
        wandb.log({
            f"{dataset}_confusion_matrix_joint": wandb.Image(figure_joint),
        })
    
    # Log hyperparameters
    wandb.config.update(args)

if __name__ == "__main__":
    main()