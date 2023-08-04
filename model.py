import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torchmetrics as tm
from transformers import AutoModelForSequenceClassification
import pytorch_lightning as pl
from data import *


class FakeTrueNewsClfModel(pl.LightningModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        class_names: list=['fake', 'real'],
        learning_rate: float=1e-3, 
        batch_size: int=32, 
        max_sequence_length: int=256,
        num_workers: int=2
    ):
        super().__init__()
        self.num_classes = len(class_names)
        self.class_names = class_names

        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        self.tf_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            problem_type='single_label_classification',
            num_labels=self.num_classes,
            output_attentions=False,
            output_hidden_states=False
        )

        self.base_grad(requires_grad=False)

        self.lr = learning_rate

        self.batch_size = batch_size

        self.train_acc = tm.Accuracy(task='binary', compute_on_step=False)
        self.val_acc = tm.Accuracy(task='binary', compute_on_step=False)
        self.test_acc = tm.Accuracy(task='binary', compute_on_step=False)
        
        self.test_cm = tm.ConfusionMatrix(task='binary', compute_on_step=False)

        self.max_seq_len = max_sequence_length

        self.num_workers = num_workers
    
    def base_grad(self, requires_grad: bool):
        for param in self.tf_model.__getattr__('distilbert').parameters():
            param.requires_grad = requires_grad

    def forward(self, batch):
        out = self.tf_model(**batch)

        return out.loss, out.logits, out.attentions
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], 
            lr=self.lr, eps=1e-08)
    
    def train_dataloader(self):
        dataset = FakeTrueNewsDataset(
            self.train_df, max_sequence_length=self.max_seq_len)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return loader

    def val_dataloader(self):
        dataset = FakeTrueNewsDataset(
            self.valid_df, max_sequence_length=self.max_seq_len)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return loader

    def test_dataloader(self):
        dataset = FakeTrueNewsDataset(
            self.test_df, max_sequence_length=self.max_seq_len)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return loader
        
    def training_step(self, batch, batch_idx):
        loss, logit, _ = self.forward(batch)
        
        logit = torch.argmax(logit, dim=1)
        
        self.train_acc.update(logit, batch['labels'])
        self.log_dict(
            {
                'train/loss': loss, 
                'train/acc': self.train_acc, 
            }, 
            on_epoch=True, 
            on_step=False,
            prog_bar=True
        )
        
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()

        self.val_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        loss, logit, _ = self.forward(batch)
        
        logit = torch.argmax(logit, dim=1)

        self.val_acc.update(logit, batch['labels'])
        self.log_dict(
            {
                'val/loss': loss, 
                'val/acc': self.val_acc
            },
            prog_bar=True
        )

    def plot_confusion_matrix(self, df):
        plt.figure(figsize=(4,2))
        ax = sns.heatmap(df, annot=True, cmap='magma', fmt='')
        ax.set_title(f'Confusion Matrix (Epoch {self.current_epoch+1})')
        ax.set_ylabel('True labels')
        ax.set_xlabel('Predicted labels')
        plt.show()

    def test_step(self, batch, batch_idx):
        loss, logit, _ = self.forward(batch)
        
        logit = torch.argmax(logit, dim=1)

        self.test_acc.update(logit, batch['labels'])
        self.log_dict(
            {
                'test/loss': loss, 
                'test/acc': self.test_acc
            },
            prog_bar=True
        )

        self.test_cm.update(logit, batch['labels'])

    def on_test_epoch_end(self):
        self.plot_confusion_matrix(
            pd.DataFrame(self.test_cm.compute().detach().cpu().numpy().astype(int)))

    def predict_step(self, batch, batch_idx):
        return self(batch)