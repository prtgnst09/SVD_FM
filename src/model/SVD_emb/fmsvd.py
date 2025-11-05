from typing import Any
import torch
import torch.nn as nn
from src.model.SVD_emb.layers import MLP, FeatureEmbedding, FM_Linear, FM_Interaction

# lightning
import pytorch_lightning as pl

class FMSVD(pl.LightningModule):
    def __init__(self, args, field_dims):
        super(FMSVD, self).__init__()
        self.lr = args.lr
        self.args = args
        self.sig = nn.Sigmoid()
        self.last_linear = nn.Linear(2, 1)
        if args.model_type=='fm':
            self.embedding = FeatureEmbedding(args, field_dims)
        self.linear = FM_Linear(args, field_dims)
        self.interaction = FM_Interaction(args)
        self.bceloss = nn.BCEWithLogitsLoss()


    def loss(self, y_pred, y_true,c_values):
        # calculate weighted mse with l2 regularization
        bce = self.bceloss(y_pred, y_true.float())
        weighted_bce = c_values * bce
        loss_y = weighted_bce.mean()
        return loss_y
    
    def forward(self, x, emb_x, svd_emb, x_cont):
        # x: batch_size * num_features
        lin_term = self.linear(x=x, emb_x=svd_emb, x_cont=x_cont)
        inter_term, cont_emb = self.interaction(emb_x=emb_x, svd_emb=svd_emb, x_cont=x_cont)
        # to normalize lin_term and inter_term to be in the same scale
        # so that the weights can be comparable
        lin_term_sig = self.sig(lin_term)
        inter_term_sig = self.sig(inter_term)
        outs = torch.cat((lin_term_sig,inter_term_sig),1)
        x = self.last_linear(outs)
        x = x.squeeze(1)
        return x, cont_emb, lin_term, inter_term

    def training_step(self, batch, batch_idx):
        x, svd_emb, x_cont, y, c_values = batch
        emb_x = self.embedding(x)
        y_pred, _, _, _ = self.forward(x=x, emb_x=emb_x, svd_emb=svd_emb, x_cont=x_cont)
        loss_y = self.loss(y_pred, y, c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) ->Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        return optimizer