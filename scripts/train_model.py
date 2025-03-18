
from hippie.dataloading import EphysDataset, EphysDatasetLabeled, BalancedBatchSampler
from hippie.model import hippieUnimodalCVAE, hippieUnimodalEmbeddingModelCVAE
from utils import get_embeddings, generate_kfolds, make_confmat

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
# The argparse inputs are going to be the z_dim and the noise addition
parser = argparse.ArgumentParser()
parser.add_argument(
    "--z_dim",
    type=int,
    default=64,
    required=False,
)

parser.add_argument(
    '--weight-decay',
    type=float,
    default=0.01,
)

parser.add_argument(
    '--learning-rate',
    type=float,
    default=0.001,
)

parser.add_argument(
    '--beta',
    type=float,
    default=1,
)

parser.add_argument(
    '--dataset',
    type=str,
    default="cellexplorer-celltype",
)

parser.add_argument('--upload-model', action='store_true')

wandb_tag = "no_curr_sup_pretrain_data"
args = parser.parse_args()
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
limit_train_batches = None
limit_val_batches = None
epochs = -1

normalize = True

project = "HIPPIE final benchmarks w finetune without labels"
FINETUNE_WITHOUT_LABELS = True

# Seed
torch.manual_seed(42)

# autocurated_wf = pd.read_csv("datasets/autocurated/waveforms.csv")
# autocurated_isi = pd.read_csv("datasets/autocurated/isi_dist.csv")
# manual_templates = pd.read_csv("datasets/unlabeled_templates/waveforms.csv")
# manual_isi = pd.read_csv("datasets/unlabeled_templates/isi_dist.csv")

# dataset_files = {
#     "a1data-remove-undef": 1,
#     "autocurated": 2,
#     "unlabeled_templates": 2,
#     "cellexplorer": 3,
#     "cellexplorer-area": 3,
#     "jianing-layer-cell-fig3": 4,
#     "jianing-area": 4,
#     "cellexplorer-physmap-test": 3,
#     "visual-allen": 3,
#     "mouse-slice-area": 2,
# }
# all_dataset_files = dataset_files.copy()
# all_dataset_files.update(
#     {
#         "biccuculine-dataset": 2,
#         "cluster-distribution-cellline": 2,
#         "chip19894": 2,
#     }
# )

dataset_files = {
    "extracellular-mouse-a1": 1,
    "cellexplorer-celltype": 3,
    "cellexplorer-area": 3,
    "justacellular-mouse-s1-celltype": 4,
    "juxtacellular-mouse-s1-area": 4,
    "allenscope-neuropixel": 3,
    "neonatal-mouse-brain-slice": 2,
}

all_dataset_files = dataset_files.copy()

num_sources = max(all_dataset_files.values()) + 1

all_waveforms = []
all_isi = []
labels = []
datasets_wf = []
datasets_isi = []

if "justacellular" in args.dataset:
    dataset_files.pop("justacellular-mouse-s1-celltype")
    dataset_files.pop("juxtacellular-mouse-s1-area")

if "cellexplorer" in args.dataset:
    dataset_files.pop("cellexplorer-celltype")
    dataset_files.pop("cellexplorer-area")

for folder in dataset_files:
    if folder != args.dataset: # dont use the dataset we test on for pretraining
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
prop = 0.8
indices = list(range(len(all_waveform_dataset)))
train_indices, test_indices = random_split(indices, [int(prop * len(indices)), len(indices) - int(prop * len(indices))])

# Split the datasets using the defined indices
train_dataset_wave = torch.utils.data.Subset(all_waveform_dataset, train_indices)
test_dataset_wave = torch.utils.data.Subset(all_waveform_dataset, test_indices)

train_dataset_time = torch.utils.data.Subset(all_isi_dataset, train_indices)
test_dataset_time = torch.utils.data.Subset(all_isi_dataset, test_indices)

train_loader_wave = torch.utils.data.DataLoader(
    train_dataset_wave, batch_size=512, shuffle=True
)
test_loader_wave = torch.utils.data.DataLoader(
    test_dataset_wave, batch_size=512, shuffle=False
)
train_loader_time = torch.utils.data.DataLoader(
    train_dataset_time, batch_size=512, shuffle=True
)
test_loader_time = torch.utils.data.DataLoader(
    test_dataset_time, batch_size=512, shuffle=False
)

wave_model = hippieUnimodalCVAE(z_dim=args.z_dim, output_size=50, class_hidden_dim=5, num_sources=num_sources, num_classes=5)
time_model = hippieUnimodalCVAE(z_dim=args.z_dim, output_size=100, class_hidden_dim=5, num_sources=num_sources, num_classes=5)

wave_model = hippieUnimodalEmbeddingModelCVAE(wave_model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
time_model = hippieUnimodalEmbeddingModelCVAE(time_model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

wave_modelcheckpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_loss", save_top_k=1, mode="min"
)
time_modelcheckpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_loss", save_top_k=1, mode="min"
)
early_stop_wave = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=30, mode="min"
)
early_stop_time = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=30, mode="min"
)


wandb_logger1 = pl.loggers.WandbLogger(
    project=project,
    name=f"{wandb_tag}{args.dataset}_wave_model_{args.z_dim}",
)
trainer_wave = pl.Trainer(
    max_epochs=150,
    accelerator=accelerator,
    logger=wandb_logger1,
    callbacks=[wave_modelcheckpoint, early_stop_wave],
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
)
trainer_wave.fit(wave_model, train_loader_wave, test_loader_wave)
wandb_logger2 = pl.loggers.WandbLogger(
    project=project,
    name=f"{wandb_tag}{args.dataset}_time_model_{args.z_dim}",
)
trainer_time = pl.Trainer(
    max_epochs=150,
    accelerator=accelerator,
    logger=wandb_logger2,
    callbacks=[time_modelcheckpoint, early_stop_time],
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    gradient_clip_val=1,
)
trainer_time.fit(time_model, train_loader_time, test_loader_time)

wave_path = wave_modelcheckpoint.best_model_path
time_path = time_modelcheckpoint.best_model_path
wave_model.load_state_dict(torch.load(wave_path)["state_dict"])
time_model.load_state_dict(torch.load(time_path)["state_dict"])

# Get the embeddings on the fine tuning dataset before fine tuning 
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
    prop = 0.1
    indices = list(range(len(finetune_dataset_wave)))
    
    if os.path.exists(f"datasets/{args.dataset}/metadata.csv") and "chip" in args.dataset:
        metadata = pd.read_csv(f"datasets/{args.dataset}/metadata.csv")
        metadata['datetime'] = pd.to_datetime(metadata['datetime']).dt.time

        first_times = metadata['datetime'].sort_values().unique()[:10]
        train_indices = metadata[metadata['datetime'].isin(first_times)].index.tolist()
        test_indices = metadata[~metadata['datetime'].isin(first_times)].index.tolist()
    else:
        train_indices, test_indices = random_split(indices, [int(prop * len(indices)), len(indices) - int(prop * len(indices))])

    wave_model = hippieUnimodalEmbeddingModelCVAE(wave_model.model, learning_rate=(1/10)*args.learning_rate, weight_decay=args.weight_decay)
    time_model = hippieUnimodalEmbeddingModelCVAE(time_model.model, learning_rate=(1/10)*args.learning_rate, weight_decay=args.weight_decay)
    # Split the datasets using the defined indices
    train_finetune_dataset = torch.utils.data.Subset(finetune_dataset_wave, train_indices)
    test_finetune_dataset = torch.utils.data.Subset(finetune_dataset_wave, test_indices)

    train_finetune_loader_wave = torch.utils.data.DataLoader(
        train_finetune_dataset, batch_size=512, shuffle=False
    )
    test_finetune_loader_wave = torch.utils.data.DataLoader(
        test_finetune_dataset, batch_size=512, shuffle=False
    )

    # get time datasets 
    train_finetune_dataset_time = torch.utils.data.Subset(finetune_dataset_time, train_indices)
    test_finetune_dataset_time = torch.utils.data.Subset(finetune_dataset_time, test_indices)
    train_finetune_loader_time = torch.utils.data.DataLoader(
        train_finetune_dataset_time, batch_size=512, shuffle=False
    )
    test_finetune_loader_time = torch.utils.data.DataLoader(
        test_finetune_dataset_time, batch_size=512, shuffle=False
    )

    # finetune the model 
    trainer_wave = pl.Trainer(
        max_epochs=50,
        accelerator=accelerator,
        logger=wandb_logger1,
        callbacks=[wave_modelcheckpoint, early_stop_wave],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    trainer_wave.fit(wave_model, train_finetune_loader_wave, test_finetune_loader_wave)
    wandb_logger2 = pl.loggers.WandbLogger(
        project=project,
        name=f"{wandb_tag}{args.dataset}_time_model_{args.z_dim}",
    )
    trainer_time = pl.Trainer(
        max_epochs=50,
        accelerator=accelerator,
        logger=wandb_logger2,
        callbacks=[time_modelcheckpoint, early_stop_time],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=1,
    )
    trainer_time.fit(time_model, train_finetune_loader_time, test_finetune_loader_time)

    (
        finetune_embeddings_wave,
        finetune_embeddings_time,
        finetune_joint_embeddings,
    ) = get_embeddings(train_finetune_loader_wave, train_finetune_loader_time, wave_model, time_model)
else:
    finetune_loader_wave = torch.utils.data.DataLoader(
        finetune_dataset_wave, batch_size=512, shuffle=False
    )

    finetune_loader_time = torch.utils.data.DataLoader(
        finetune_dataset_time, batch_size=512, shuffle=False
    )

    (
        finetune_embeddings_wave,
        finetune_embeddings_time,
        finetune_joint_embeddings,
    ) = get_embeddings(finetune_loader_wave, finetune_loader_time, wave_model, time_model)

# upload pretraining embeddings with "train" or "val" as the "split" column
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

dataset = args.dataset
folds, all_supervised_wf, all_supervised_isi, all_supervised_labels = generate_kfolds(dataset)

total_best_confmat_isi = []
total_best_confmat_waveform = []
total_best_confmat_joint = []

total_best_balanced_accuracy_isi = []
total_best_balanced_accuracy_waveform = []
total_best_balanced_accuracy_joint = []

wf_dfs = {"pred": [], "true": [], "fold": []}
isi_dfs = {"pred": [], "true": [], "fold": []}
joint_dfs = {"pred": [], "true": [], "fold": []}

wf_embeddings = {"embeddings": [], "fold": []}
isi_embeddings = {"embeddings": [], "fold": []}
joint_embeddings = {"embeddings": [], "fold": []}

for fold_idx, (wf_train, wf_val, isi_train, isi_val, label_train, label_val, le) in enumerate(folds):
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

    train_sampler = BalancedBatchSampler(
        dataset_train_wave, label_train, 
    )
    train_loader_wave = torch.utils.data.DataLoader(
        dataset_train_wave, batch_size=64, sampler=train_sampler, num_workers=4
    )

    test_loader_wave = torch.utils.data.DataLoader(
        dataset_val_wave, batch_size=64, shuffle=False, num_workers=4
    )

    train_loader_time = torch.utils.data.DataLoader(
        dataset_train_time, batch_size=64, sampler=train_sampler, num_workers=4
    )
    test_loader_time = torch.utils.data.DataLoader(
        dataset_val_time, batch_size=64, shuffle=False, num_workers=4
    )

    # # load the models
    wave_model = hippieUnimodalCVAE(z_dim=args.z_dim, output_size=50, class_hidden_dim=5, num_sources=num_sources, num_classes=num_class_labels)
    time_model = hippieUnimodalCVAE(z_dim=args.z_dim, output_size=100, class_hidden_dim=5, num_sources=num_sources, num_classes=num_class_labels)
    # # #Train models

    wave_seq = torch.load(wave_path)
    wave_seq["state_dict"].pop("model.class_embedding.weight")
    wave_model = hippieUnimodalEmbeddingModelCVAE(wave_model, learning_rate=(1/10)*args.learning_rate, weight_decay=args.weight_decay)
    wave_model.load_state_dict(wave_seq["state_dict"], strict=False)
    
    time_seq = torch.load(time_path)
    time_seq["state_dict"].pop("model.class_embedding.weight")
    time_model = hippieUnimodalEmbeddingModelCVAE(time_model, learning_rate=(1/10)*args.learning_rate, weight_decay=args.weight_decay)
    time_model.load_state_dict(time_seq["state_dict"], strict=False)

    wave_modelcheckpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min"
    )
    time_modelcheckpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min"
    )
    early_stop_wave = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, mode="min"
    )
    early_stop_time = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, mode="min"
    )

    wandb_logger1 = pl.loggers.WandbLogger(
        project=project,
        name=f"{wandb_tag}{args.dataset}finetune_wave_model_{dataset}",
    )
    lr_monitor_time = pl.callbacks.LearningRateMonitor(logging_interval="step")
    lr_monitor_wave = pl.callbacks.LearningRateMonitor(logging_interval="step")

    trainer_wave = pl.Trainer(
        max_epochs=50,
        accelerator=accelerator,
        logger=wandb_logger1,
        callbacks=[wave_modelcheckpoint, early_stop_wave, lr_monitor_wave],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=1,
    )
    trainer_wave.fit(wave_model, train_loader_wave, test_loader_wave)
    wandb_logger2 = pl.loggers.WandbLogger(
        project=project,
        name=f"{wandb_tag}{args.dataset}finetune_time_mode_{dataset}",
    )
    trainer_time = pl.Trainer(
        max_epochs=50,
        accelerator=accelerator,
        logger=wandb_logger2,
        callbacks=[time_modelcheckpoint, early_stop_time, lr_monitor_time],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=1,
    )
    trainer_time.fit(time_model, train_loader_time, test_loader_time)

    train_loader_wave = torch.utils.data.DataLoader(dataset_train_wave, batch_size=128)
    train_loader_time = torch.utils.data.DataLoader(dataset_train_time, batch_size=128)

    # load in best epoch after fine tuning 
    wave_path = wave_modelcheckpoint.best_model_path
    time_path = time_modelcheckpoint.best_model_path
    # log the best epoch to wandb
    wandb.log(
        {
            "best_epoch_waveform": wave_path,
            "best_epoch_time": time_path,
        }
    )

    wave_seq = torch.load(wave_path)
    wave_model.load_state_dict(wave_seq["state_dict"])
    wave_model.optimizer.load_state_dict(wave_seq["optimizer_states"][0])

    time_seq = torch.load(time_path)
    time_model.load_state_dict(time_seq["state_dict"])
    time_model.optimizer.load_state_dict(time_seq["optimizer_states"][0])

    time_model.eval()
    wave_model.eval()

    (
        waveform_embeddings_train,
        isi_dist_embeddings_train,
        joint_embeddings_train,
    ) = get_embeddings(train_loader_wave, train_loader_time, wave_model, time_model)

    (
        waveform_embeddings_test,
        isi_dist_embeddings_test,
        joint_embeddings_test,
    ) = get_embeddings(test_loader_wave, test_loader_time, wave_model, time_model)

    joint_bal_accuracy = []
    waveform_bal_accuracy = []
    isi_bal_accuracy = []
    neighbor_options = list(range(5, 20))

    for neighbor in neighbor_options:
        print("KNN with", neighbor, "neighbors")
        knn = KNeighborsClassifier(n_neighbors=neighbor)
        knn.fit(joint_embeddings_train, label_train)
        train_pred = knn.predict(joint_embeddings_train)
        pred = knn.predict(joint_embeddings_test)
        joint_bal_accuracy.append(balanced_accuracy_score(label_val, pred))

        # Waveform embeddings
        knn = KNeighborsClassifier(n_neighbors=neighbor)
        knn.fit(waveform_embeddings_train, label_train)
        # Log training results
        train_pred = knn.predict(waveform_embeddings_train)
        pred = knn.predict(waveform_embeddings_test)
        waveform_bal_accuracy.append(balanced_accuracy_score(label_val, pred))

        # for isi embeddings
        knn = KNeighborsClassifier(n_neighbors=neighbor)
        knn.fit(isi_dist_embeddings_train, label_train)
        # Log training results
        train_pred = knn.predict(isi_dist_embeddings_train)
        pred = knn.predict(isi_dist_embeddings_test)
        isi_bal_accuracy.append(balanced_accuracy_score(label_val, pred))

    total_best_balanced_accuracy_isi.append(max(isi_bal_accuracy))
    total_best_balanced_accuracy_waveform.append(max(waveform_bal_accuracy))
    total_best_balanced_accuracy_joint.append(max(joint_bal_accuracy))

    # print(f"Label names {le.classes_}")
    label_names = le.classes_

    # log the best accuracy for each modality
    best_neighbors_waveform = neighbor_options[np.argmax(waveform_bal_accuracy)]
    knn = KNeighborsClassifier(n_neighbors=best_neighbors_waveform)
    knn.fit(waveform_embeddings_train, label_train)
    pred = knn.predict(waveform_embeddings_test)
    total_best_confmat_waveform.append(confusion_matrix(label_val, pred))

    pred = le.inverse_transform(pred)
    wf_dfs["pred"].extend(pred)
    wf_dfs["true"].extend(le.inverse_transform(label_val))
    wf_dfs["fold"].extend([fold_idx] * len(label_val))

    wf_embeddings["embeddings"].extend(waveform_embeddings_test)
    wf_embeddings["fold"].extend([fold_idx] * len(label_val))

    best_neighbors_isi = neighbor_options[np.argmax(isi_bal_accuracy)]
    knn = KNeighborsClassifier(n_neighbors=best_neighbors_isi)
    knn.fit(isi_dist_embeddings_train, label_train)
    pred = knn.predict(isi_dist_embeddings_test)
    total_best_confmat_isi.append(confusion_matrix(label_val, pred))
    
    pred = le.inverse_transform(pred)
    isi_dfs["pred"].extend(pred)
    isi_dfs["true"].extend(le.inverse_transform(label_val))
    isi_dfs["fold"].extend([fold_idx] * len(label_val))

    isi_embeddings["embeddings"].extend(isi_dist_embeddings_test)
    isi_embeddings["fold"].extend([fold_idx] * len(label_val))

    best_neighbors_joint = neighbor_options[np.argmax(joint_bal_accuracy)]
    knn = KNeighborsClassifier(n_neighbors=best_neighbors_joint)
    knn.fit(joint_embeddings_train, label_train)
    pred = knn.predict(joint_embeddings_test)
    total_best_confmat_joint.append(confusion_matrix(label_val, pred))

    pred = le.inverse_transform(pred)
    joint_dfs["pred"].extend(pred)
    joint_dfs["true"].extend(le.inverse_transform(label_val))
    joint_dfs["fold"].extend([fold_idx] * len(label_val))

    joint_embeddings["embeddings"].extend(joint_embeddings_test)
    joint_embeddings["fold"].extend([fold_idx] * len(label_val))

wf_df = pd.DataFrame(wf_dfs)
isi_df = pd.DataFrame(isi_dfs)
joint_df = pd.DataFrame(joint_dfs)

wf_df.to_csv(f"{dataset}_waveform_knn.csv")
isi_df.to_csv(f"{dataset}_isi_knn.csv")
wf_df.to_csv(f"{dataset}_joint_knn.csv")

wandb.log_artifact(f"{dataset}_waveform_knn.csv", name=f"{dataset}_waveform_knn.csv", type=f"{dataset}_waveform_knn.csv")
wandb.log_artifact(f"{dataset}_isi_knn.csv", name=f"{dataset}_isi_knn.csv", type=f"{dataset}_isi_knn.csv")
wandb.log_artifact(f"{dataset}_joint_knn.csv", name=f"{dataset}_joint_knn.csv", type=f"{dataset}_joint_knn.csv")

all_wf_dataloader = torch.utils.data.DataLoader(
    EphysDatasetLabeled(all_supervised_wf, all_supervised_isi, np.vstack((all_supervised_labels, np.ones_like(all_supervised_labels) * all_dataset_files[dataset])).T, mode="wave", normalize=False),
)

all_isi_dataloader = torch.utils.data.DataLoader(
    EphysDatasetLabeled(all_supervised_wf, all_supervised_isi, np.vstack((all_supervised_labels, np.ones_like(all_supervised_labels) * all_dataset_files[dataset])).T, mode="time", normalize=False),
)

wf_embeddings, isi_embeddings, joint_embeddings = get_embeddings(all_wf_dataloader, all_isi_dataloader, wave_model, time_model)

wf_embeddings_df = pd.DataFrame(wf_embeddings)
isi_embeddings_df = pd.DataFrame(isi_embeddings)
joint_embeddings_df = pd.DataFrame(joint_embeddings)

wf_embeddings_df["label"] = le.inverse_transform(all_supervised_labels)
isi_embeddings_df["label"] = le.inverse_transform(all_supervised_labels)
joint_embeddings_df["label"] = le.inverse_transform(all_supervised_labels)

wf_embeddings_df.to_csv(f"{dataset}_waveform_embeddings.csv")
isi_embeddings_df.to_csv(f"{dataset}_isi_embeddings.csv")
joint_embeddings_df.to_csv(f"{dataset}_joint_embeddings.csv")

wandb.log_artifact(f"{dataset}_waveform_embeddings.csv", name=f"{dataset}_waveform_embeddings.csv", type=f"{dataset}_waveform_embeddings.csv")
wandb.log_artifact(f"{dataset}_isi_embeddings.csv", name=f"{dataset}_isi_embeddings.csv", type=f"{dataset}_isi_embeddings.csv")
wandb.log_artifact(f"{dataset}_joint_embeddings.csv", name=f"{dataset}_joint_embeddings.csv", type=f"{dataset}_joint_embeddings.csv")

# log the last split to wandb
if args.upload_model:
    wandb.log_artifact(wave_path, name=f'wave_model_ft_d{args.dataset}_z{args.z_dim}_lr{args.learning_rate}.pt', type='model')
    wandb.log_artifact(time_path, name=f'time_model_ft_d{args.dataset}_z{args.z_dim}_lr{args.learning_rate}.pt', type='model')

# Log the best balanced accuracy for each modality
wandb.log(
    {
        "best_balanced_accuracy_waveform_mean": np.mean(total_best_balanced_accuracy_waveform),
        "best_balanced_accuracy_isi_mean": np.mean(total_best_balanced_accuracy_isi),
        "best_balanced_accuracy_joint_mean": np.mean(total_best_balanced_accuracy_joint),
    }
)

# Log the sum of confusion matrices
sum_confmat_waveform = np.sum(total_best_confmat_waveform, axis=0)
sum_confmat_isi = np.sum(total_best_confmat_isi, axis=0)
sum_confmat_joint = np.sum(total_best_confmat_joint, axis=0)

# make the confusion matrix imagse 
figure_waveform = make_confmat(sum_confmat_waveform, label_names, best_neighbors_waveform)
figure_isi = make_confmat(sum_confmat_isi, label_names, best_neighbors_isi)
figure_joint = make_confmat(sum_confmat_joint, label_names, best_neighbors_joint)

# log as images to wandb 
wandb.log(
    {
        f"{dataset}_confusion_matrix_waveform_cv": wandb.Image(figure_waveform),
        f"{dataset}_confusion_matrix_isi_cv": wandb.Image(figure_isi),
        f"{dataset}_confusion_matrix_joint_cv": wandb.Image(figure_joint),
    }
)

# log all the hyperparameters
wandb.config.update(args)
