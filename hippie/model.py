import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from backbones import ResNet18Enc, ResNet18Dec
from pytorch_lightning.utilities import grad_norm
from torch.nn.functional import normalize
from optimizers import AdamWScheduleFree



class hippieUnimodalCVAE(nn.Module):
    def __init__(self, z_dim, output_size, class_hidden_dim, num_sources, num_classes):
        super().__init__()
        self.z_dim = z_dim
        self.class_hidden_dim = class_hidden_dim
        self.num_sources = num_sources
        self.num_classes = num_classes

        self.encoder = ResNet18Enc(z_dim=z_dim)  # Assuming this already includes appropriate normalization
        self.encoder_fc = nn.Sequential(
            nn.Linear(z_dim * 2 + class_hidden_dim * 2, z_dim * 2),
            nn.BatchNorm1d(z_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim * 2, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.LeakyReLU(0.2)
        )

        self.source_embedding = nn.Embedding(num_sources, class_hidden_dim)
        self.class_embedding = nn.Embedding(num_classes, class_hidden_dim)
        self.z_mean = nn.Linear(z_dim, z_dim)
        self.z_log_var = nn.Linear(z_dim, z_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(z_dim + class_hidden_dim * 2, z_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.BatchNorm1d(z_dim * 2),
            nn.LeakyReLU(0.2),
        )
        self.decoder = ResNet18Dec(z_dim=z_dim, output_size=output_size)  # Assuming this includes appropriate normalization

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, source_emb, class_emb):
        x = self.encoder(x)
        x = torch.cat([x, source_emb, class_emb], dim=1)
        x = self.encoder_fc(x)
        mu = self.z_mean(x)
        logvar = self.z_log_var(x)
        return x, mu, logvar

    def decode(self, z, source_emb, class_emb):
        z = torch.cat([z, source_emb, class_emb], dim=1)
        z = self.decoder_fc(z)
        return self.decoder(z)

    def forward(self, data, source_labels, class_labels=None):
        source_emb = self.source_embedding(source_labels)
        class_emb = self.class_embedding(class_labels) if class_labels is not None else torch.zeros_like(source_emb)

        encoded, mu, logvar = self.encode(data, source_emb, class_emb)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, source_emb, class_emb)

        return encoded, mu, logvar, decoded


class hippieUnimodalEmbeddingModelCVAE(nn.Module):
    def __init__(
        self, base_model, alpha_max=0.5, learning_rate=0.01, weight_decay=0.01, beta=1,
    ):
        super().__init__()
        self.alpha_max = alpha_max
        self.beta = beta

    def training_step(self, batch, batch_idx):
        data, labels = batch
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, dec = self.model(data, source_labels=source_labels, class_labels=class_labels)
        else:
            enc, zmean, zlogvar, dec = self.model(data, source_labels=labels)
        
        mse_loss = F.mse_loss(data, dec)
        kl_loss = -0.5 * torch.sum(1 + zlogvar - zmean.pow(2) - torch.exp(zlogvar), axis=1)
        
        total_epochs = self.trainer.max_epochs
        current_epoch = self.current_epoch
                
        loss = mse_loss + self.beta * kl_loss.mean()
        
        self.log("train_loss", loss)
        self.log("train_mse_loss", mse_loss)
        self.log("train_kl_loss", kl_loss.mean())
        self.train_loss.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, dec = self.model(data, source_labels=source_labels, class_labels=class_labels)
        else:
            enc, zmean, zlogvar, dec = self.model(data, source_labels=labels)
        
        mse_loss = F.mse_loss(data, dec)
        kl_loss = -0.5 * torch.sum(1 + zlogvar - zmean.pow(2) - torch.exp(zlogvar), axis=1)
        
        total_epochs = self.trainer.max_epochs
        current_epoch = self.current_epoch
        
        loss = mse_loss + self.beta * kl_loss.mean()

        self.val_loss.append(loss.item())
        self.log("val_loss", loss)
        self.log("val_mse_loss", mse_loss)
        self.log("val_kl_loss", kl_loss.mean())

        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = sum(self.val_loss) / len(self.val_loss)
        print(f"Average validation loss is {avg_loss:.2f}")
        self.val_loss = []
        
    def on_train_epoch_end(self):
        avg_loss = sum(self.train_loss) / len(self.train_loss)
        print(f"Average training loss is {avg_loss:.2f}")
        self.train_loss = []

    def configure_optimizers(self):
        return self.optimizer
    
    def forward(self, batch):
        data, labels = batch
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, dec = self.model(data, source_labels=source_labels, class_labels=class_labels)
        else:
            enc, zmean, zlogvar, dec = self.model(data, source_labels=labels)

        return enc, zmean, zlogvar, dec
    
class MultiModalCVAE(nn.Module):
    def __init__(self, z_dim, output_size_wave, output_size_isi, class_hidden_dim, num_sources, num_classes):
        super().__init__()
        self.z_dim = z_dim
        self.class_hidden_dim = class_hidden_dim
        self.num_sources = num_sources
        self.num_classes = num_classes

        # Separate encoders for each modality
        self.encoder_mod1 = ResNet18Enc(z_dim=z_dim)
        self.encoder_mod2 = ResNet18Enc(z_dim=z_dim)
        
        # Shared fusion layers
        self.fusion_encoder = nn.Sequential(
            nn.Linear((z_dim * 2) * 2 + class_hidden_dim * 2, z_dim * 2),
            nn.BatchNorm1d(z_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim * 2, z_dim),
            # nn.BatchNorm1d(z_dim),
            # nn.LeakyReLU(0.2)
        )

        self.source_embedding = nn.Embedding(num_sources, class_hidden_dim)
        self.class_embedding = nn.Embedding(num_classes, class_hidden_dim)
        
        # Single set of latent variables for joint distribution
        self.z_mean = nn.Linear(z_dim, z_dim)
        self.z_log_var = nn.Linear(z_dim, z_dim)

        # Separate decoders for each modality
        self.decoder_fc_mod1 = nn.Sequential(
            nn.Linear(z_dim + class_hidden_dim * 2, z_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.BatchNorm1d(z_dim * 2),
            nn.LeakyReLU(0.2),
        )
        self.decoder_fc_mod2 = nn.Sequential(
            nn.Linear(z_dim + class_hidden_dim * 2, z_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.BatchNorm1d(z_dim * 2),
            nn.LeakyReLU(0.2),
        )
        
        self.decoder_mod1 = ResNet18Dec(z_dim=z_dim, output_size=output_size_wave)
        self.decoder_mod2 = ResNet18Dec(z_dim=z_dim, output_size=output_size_isi)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x1, x2, source_emb, class_emb):
        h1 = self.encoder_mod1(x1)
        h2 = self.encoder_mod2(x2)
        
        h = torch.cat([h1, h2, source_emb, class_emb], dim=1)
        h = self.fusion_encoder(h)
        return h, self.z_mean(h), self.z_log_var(h)

    def decode(self, z, source_emb, class_emb):
        # Prepare latent vector for each decoder
        z1 = torch.cat([z, source_emb, class_emb], dim=1)
        z2 = torch.cat([z, source_emb, class_emb], dim=1)
        
        # Decode both modalities
        z1 = self.decoder_fc_mod1(z1)
        z2 = self.decoder_fc_mod2(z2)
        
        recon1 = self.decoder_mod1(z1)
        recon2 = self.decoder_mod2(z2)
        
        return recon1, recon2

    def forward(self, data1, data2, source_labels, class_labels=None):
        source_emb = self.source_embedding(source_labels)
        class_emb = self.class_embedding(class_labels) if class_labels is not None else torch.zeros_like(source_emb)

        encoded, mu, logvar = self.encode(data1, data2, source_emb, class_emb)
        z = self.reparameterize(mu, logvar)
        decoded1, decoded2 = self.decode(z, source_emb, class_emb)

        return encoded, mu, logvar, decoded1, decoded2

class MultiModalCVAETrainModule(pl.LightningModule):
    def __init__(
        self, base_model, alpha_max=0.5, learning_rate=0.01, weight_decay=0.01, beta=1,
        mod1_weight=1.0, mod2_weight=1.0
    ):
        super().__init__()
        self.model = base_model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.mse_loss = nn.MSELoss()
        self.val_loss = []
        self.train_loss = []
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.mod1_weight = mod1_weight
        self.mod2_weight = mod2_weight
        self.alpha_max = alpha_max
        self.beta = beta

    def training_step(self, batch, batch_idx):
        data1, data2, labels = batch
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, dec1, dec2 = self.model(
                data1, data2, source_labels=source_labels, class_labels=class_labels
            )
        else:
            enc, zmean, zlogvar, dec1, dec2 = self.model(data1, data2, source_labels=labels)
        
        # Separate reconstruction losses for each modality
        mse_loss1 = F.mse_loss(data1, dec1)
        mse_loss2 = F.mse_loss(data2, dec2)
        
        # Weighted combination of reconstruction losses
        mse_loss = self.mod1_weight * mse_loss1 + self.mod2_weight * mse_loss2
        
        # Single KL loss for joint latent space
        kl_loss = -0.5 * torch.sum(1 + zlogvar - zmean.pow(2) - torch.exp(zlogvar), axis=1)
        
        total_loss = mse_loss + self.beta * kl_loss.mean()
        
        self.log("train_loss", total_loss)
        self.log("train_mse_loss1", mse_loss1)
        self.log("train_mse_loss2", mse_loss2)
        self.log("train_kl_loss", kl_loss.mean())
        self.train_loss.append(total_loss.item())
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        data1, data2, labels = batch
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, dec1, dec2 = self.model(
                data1, data2, source_labels=source_labels, class_labels=class_labels
            )
        else:
            enc, zmean, zlogvar, dec1, dec2 = self.model(data1, data2, source_labels=labels)
        
        mse_loss1 = F.mse_loss(data1, dec1)
        mse_loss2 = F.mse_loss(data2, dec2)
        mse_loss = self.mod1_weight * mse_loss1 + self.mod2_weight * mse_loss2
        
        kl_loss = -0.5 * torch.sum(1 + zlogvar - zmean.pow(2) - torch.exp(zlogvar), axis=1)
        
        loss = mse_loss + self.beta * kl_loss.mean()

        self.val_loss.append(loss.item())
        self.log("val_loss", loss)
        self.log("val_mse_loss1", mse_loss1)
        self.log("val_mse_loss2", mse_loss2)
        self.log("val_kl_loss", kl_loss.mean())

        return loss

    def forward(self, batch):
        data1, data2, labels = batch
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, dec1, dec2 = self.model(
                data1, data2, source_labels=source_labels, class_labels=class_labels
            )
        else:
            enc, zmean, zlogvar, dec1, dec2 = self.model(data1, data2, source_labels=labels)

        return enc, zmean, zlogvar, dec1, dec2

    def on_validation_epoch_end(self):
        avg_loss = sum(self.val_loss) / len(self.val_loss)
        print(f"Average validation loss is {avg_loss:.2f}")
        self.val_loss = []
        
    def on_train_epoch_end(self):
        avg_loss = sum(self.train_loss) / len(self.train_loss)
        print(f"Average training loss is {avg_loss:.2f}")
        self.train_loss = []

    def configure_optimizers(self):
        return self.optimizer
