import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any

import torch
from src.model.SVD_emb.fmsvd import FMSVD
from src.model.SVD_emb.layers import FeatureEmbedding, FeatureEmbedding, FM_Linear, MLP
#from src.util.scaler import StandardScaler



class DeepFMSVD(pl.LightningModule):
    def __init__(self, args,field_dims):
        super(DeepFMSVD, self).__init__()
        self.args = args
        self.linear = FM_Linear(args, field_dims)
        self.fm = FMSVD(args, field_dims)
        self.embedding = FeatureEmbedding(args, field_dims)

        self.embed_output_dim = (len(field_dims))* args.emb_dim + 2*args.emb_dim+(args.cont_dims-2*args.num_eigenvector)*args.emb_dim
        self.mlp = MLP(args, self.embed_output_dim)
        self.bceloss = nn.BCEWithLogitsLoss() # since bcewith logits is used, no need to add sigmoid layer in the end
        self.lr = args.lr
        self.field_dims = field_dims
        self.sig = nn.Sigmoid()
        self.lastlinear = nn.Linear(3,1)

    def mse(self, y_pred, y_true):
        return self.bceloss(y_pred, y_true.float())

    def deep_part(self, x):
        return self.mlp(x)
    
    def loss(self, y_pred, y_true, c_values):
        bce = self.bceloss(y_pred, y_true.float())
        weighted_bce = c_values * bce
        loss_y = weighted_bce.mean()
        return loss_y
    

    def forward(self, x, embed_x, svd_emb, x_cont):
        # FM part, here, x_hat means another arbritary input of data, for combining the results. 
        _, cont_emb, lin_term, inter_term = self.fm(x, embed_x, svd_emb, x_cont)
        user_emb = svd_emb[:, :self.args.num_eigenvector]
        item_emb = svd_emb[:, self.args.num_eigenvector:]
        
        embed_x = torch.cat((embed_x, cont_emb), 1)
        feature_number = embed_x.shape[1]
        
        # to make embed_x to batch_size * (num_features*embedding_dim)
        embed_x = embed_x.reshape(-1, feature_number*self.args.emb_dim)
        
        new_x = torch.cat((embed_x, user_emb, item_emb),1)
        # new_x = torch.cat((new_x, item_emb),1)
        deep_part = self.mlp(new_x)
        
        # Deep part
        lin_term_sig = self.sig(lin_term)
        inter_term_sig = self.sig(inter_term)
        deep_part_sig = self.sig(deep_part)

        outs = torch.cat((lin_term_sig, inter_term_sig, deep_part_sig), 1)
        y_pred = self.lastlinear(outs).squeeze(1)

        return y_pred

    def training_step(self, batch):
        x, svd_emb, x_cont, y, c_values = batch
        embed_x = self.embedding(x)
        y_pred = self.forward(x, embed_x, svd_emb, x_cont)
        loss_y = self.loss(y_pred, y, c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        return optimizer