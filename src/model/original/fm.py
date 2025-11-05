from typing import Any
import torch
import torch.nn as nn
from src.model.original.layers import FeatureEmbedding, FM_Linear, FM_Interaction

import pytorch_lightning as pl
from itertools import chain

class FM(pl.LightningModule):
    def __init__(self, args, field_dims):
        super(FM, self).__init__()

        if args.model_type=='fm':
            self.embedding = FeatureEmbedding(args, field_dims)
        self.linear = FM_Linear(args, field_dims)
        self.interaction = FM_Interaction(args)
        self.bceloss = nn.BCEWithLogitsLoss() # since BCEWith logits is used, we don't need to add sigmoid layer in the end
        self.lr = args.lr
        self.args = args
        self.sig = nn.Sigmoid()
        self.last_linear = nn.Linear(2,1)

    def loss(self, y_pred, y_true, c_values):
        # calculate weighted bce with l2 regularization
        bce = self.bceloss(y_pred, y_true.float())
        weighted_bce = c_values * bce
        loss_y = weighted_bce.mean()
        return loss_y 
    
    def forward(self, x, emb_x, x_cont):
        # FM part loss with interaction terms
        # x: batch_size * num_features
        lin_term = self.linear(x=x, x_cont=x_cont)
        inter_term, cont_emb = self.interaction(emb_x, x_cont)
        lin_term_sig = self.sig(lin_term)
        inter_term_sig = self.sig(inter_term)
        outs = torch.cat((lin_term_sig, inter_term_sig), 1)
        y_pred = self.last_linear(outs)
        y_pred = y_pred.squeeze(1)
            
        return y_pred, cont_emb, lin_term, inter_term
    
    def training_step(self, batch, batch_idx):
        x, x_cont, y, c_values = batch
        embed_x = self.embedding(x)
        y_pred, _, _, _ = self.forward(x, embed_x, x_cont)
        loss_y = self.loss(y_pred, y, c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        return optimizer