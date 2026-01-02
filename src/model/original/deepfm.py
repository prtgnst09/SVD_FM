import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any

import torch
from src.model.original.fm import FM
from src.model.original.layers import FeatureEmbedding, FeatureEmbedding, FM_Linear, MLP


class DeepFM(pl.LightningModule):
    def __init__(self, args, field_dims):
        super(DeepFM, self).__init__()
        self.args = args
        self.lr = args.lr
        self.field_dims = field_dims
        self.fm = FM(args, field_dims)
        self.embedding = FeatureEmbedding(args, field_dims)
        self.embed_output_dim = len(field_dims) * args.emb_dim + args.cont_dims * args.emb_dim
        self.mlp = MLP(args, self.embed_output_dim)
        self.bceloss = nn.BCEWithLogitsLoss()

        self.sig = nn.Sigmoid()
        self.lastlinear = nn.Linear(3,1)
    
    def deep_part(self, x):
        return self.mlp(x)

    def loss(self, y_pred, y_true, c_values):
        mse = self.bceloss(y_pred, y_true.float())
        weighted_bce = c_values * mse
        loss_y = weighted_bce.mean()
        return loss_y
    
    def forward(self, x, emb_x, x_cont):
        # FM part, here, x_hat means another arbritary input of data, for combining the results. 
        _, cont_emb, lin_term, inter_term = self.fm.forward(x, emb_x, x_cont)
        
        if cont_emb is not None:
            emb_x = torch.cat((emb_x, cont_emb), 1)
        feature_number = emb_x.shape[1]
        emb_x = emb_x.view(-1, feature_number * self.args.emb_dim)

        new_x = emb_x
        mlp_x = self.mlp(new_x)

        lin_term_sig = self.sig(lin_term)
        inter_term_sig = self.sig(inter_term)
        mlp_term_sig = self.sig(mlp_x)
        outs = torch.cat((lin_term_sig, inter_term_sig, mlp_term_sig), dim=1)
        y_pred = self.lastlinear(outs).squeeze(1)

        return y_pred, cont_emb, lin_term, inter_term

    def training_step(self, batch, batch_idx):
        x, x_cont, y, c_values = batch
        embed_x = self.embedding(x)
        y_pred, _, _, _ = self.forward(x, embed_x, x_cont)
        loss_y = self.loss(y_pred, y,c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        return optimizer