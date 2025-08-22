import torch
import torch.nn as nn
import math

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.zeros(
            1, 1, max_len, d_model), requires_grad=True)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.pe.data.copy_(pe.float())
        del pe

    def forward(self, x, offset=0):
        #self.pe.pe.data.shape = torch.Size([1, 1, 1000, 32])
        return self.pe[:, :, offset:offset+x.size(2)]


class InputEncoder(nn.Module):
    def __init__(self,
                  mlp_layers,
                  lookback_len,
                  pred_len,
                  latent_channel_dim,
                  in_channels,
                  activation_fn,
                  use_stamp_data=False,
                  stamp_dim=None,
                  dropout=0.0):
        super(InputEncoder, self).__init__()

        self.in_channels = in_channels

        self.out_channels = latent_channel_dim

        self.lookback_len = lookback_len
        self.pred_len = pred_len

        self.use_stamp_data = use_stamp_data
        self.stamp_dim = stamp_dim

        if use_stamp_data:
            in_channels += stamp_dim 

        mlp_block = []
        for i in range(mlp_layers):
            if i == 0:
                in_features = self.lookback_len
            else:
                in_features = self.pred_len
            
            out_features = self.pred_len
            
            mlp_block.append(nn.Linear(in_features=in_features, out_features=out_features))
            
            if i < mlp_layers-1:
                mlp_block.append(nn.Dropout(dropout))
                mlp_block.append(activation_fn)

        self.mlp_projection = nn.Sequential(*mlp_block)


    def forward(self, x, stamp=None):

        # x.shape -> B, D, T
        B, D, T = x.size()
        if self.use_stamp_data:
            x = torch.cat([x, stamp], dim=1)
               
        # B, D, T -> B, D, T'
        skip_focal_pts = self.mlp_projection(x)
        
        mlp_edm_focal_pts = skip_focal_pts
        
        return x, mlp_edm_focal_pts, skip_focal_pts