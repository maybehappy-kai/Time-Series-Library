import torch
import torch.nn as nn
import time, math
import einops

from layers.EDM_Blocks import InputEncoder, LearnablePositionalEmbedding

class EDM(nn.Module):
    def __init__(
                self,
                lookback_len,
                out_pred_len,
                delay,
                time_delay_stride,
                layer_norm,
                latent_channel_dim,
                method, #simplex or smap
                theta,
                add_pe,
                dropout,
                activation_fn=nn.SELU(),
                dist_projection_dim=64,
                n_proj_layers=1,
                ):
        super(EDM, self).__init__()

        self.lookback_len = lookback_len
        self.out_pred_len = out_pred_len

        self.delay = delay
        self.time_delay_stride = time_delay_stride


        unfolded_len = ((lookback_len + out_pred_len) // time_delay_stride) - delay + 1
        self.unfolded_lookback_len = int((lookback_len  / (lookback_len + out_pred_len)) * unfolded_len)
        self.unfolded_pred_len = unfolded_len - self.unfolded_lookback_len


        self.latent_channel_dim = latent_channel_dim
        self.layer_norm = layer_norm
        self.method = method

        self.theta = theta
        self.activation_fn = activation_fn

        
        projection = []
        for i in range(n_proj_layers):
            if i == 0:
                projection.append(nn.Linear(self.delay, dist_projection_dim))
            else:
                projection.append(nn.Linear(dist_projection_dim, dist_projection_dim))
                projection.append(nn.Dropout(dropout))
                projection.append(self.activation_fn)

        self.projection = nn.Sequential(*projection)   
   

        if add_pe:
            self.pe = LearnablePositionalEmbedding(d_model=dist_projection_dim, max_len=max(1100,self.lookback_len+self.out_pred_len))
        else:
            self.pe = None

        self.ln1 = nn.LayerNorm(dist_projection_dim) if layer_norm else nn.Identity()
       
        self.attn_dropout = nn.Dropout(dropout)

        self.undelay = nn.Sequential(
            nn.Linear(self.delay * self.unfolded_pred_len, out_pred_len),
            nn.Dropout(dropout),
            self.activation_fn,
            nn.Linear(out_pred_len, out_pred_len)
        )
        

    def _weight_and_topk(self, A, b, focal_points):
        """
        A.shape = B,D,T,delay
        b.shape = B,D,T,edm_pred_len, delay
        F contains the focal points + the last point
        """
        B,D,F,delay = focal_points.size()
        _,_,T,_ = A.size()

        k_proj = self.projection(A)
        q_proj = self.projection(focal_points[:,:,:-1,:])

        if (self.pe is not None) and (k_proj.shape[-1] % 2 == 0):
            k_proj = k_proj + self.pe(k_proj)
            q_proj = q_proj + self.pe(q_proj, offset=k_proj.size(-2))

        k_proj = self.ln1(k_proj)
        q_proj = self.ln1(q_proj)

        scale = (1.0 / math.sqrt(k_proj.size(-1))) * self.theta
        predictions = nn.functional.scaled_dot_product_attention(q_proj, k_proj, b, dropout_p=0.1, scale=scale)

        return None, predictions, None
        
    
    def _edm_forward(self, A, b, focal_point):

        A, b, W = self._weight_and_topk(A, b, focal_point)


        if A is None and W is None:
            # When we use scaled dot product attention or sparse topk, we get the predictions directly
            return b, None
        else:
            raise NotImplementedError("Only scaled dot product attention is implemented for EDM.")
        
    def forward(self, X, focal_points):

        X = torch.cat([X, focal_points], dim=-1)

        X_td = X.unfold(-1, self.delay, self.time_delay_stride) # B,D,T -> B,D,T',delay

        focal_points = X_td[:,:,-self.unfolded_pred_len-1:,:]
        X_td = X_td[:,:,:-self.unfolded_pred_len-1,:]

        #B,D,T,delay
        A = X_td[:,:,:-1,:]
        b = X_td[:,:,1:,:]

        #focal_points have one extra point at the end
        all_focal_points = focal_points

        #pred shape B,D,F,delay
        pred, sol = self._edm_forward(A, b, all_focal_points)
        
        pred = pred.reshape(pred.size(0), pred.size(1), -1)
        pred = self.undelay(pred)

        return pred, sol


activation_fn_map = {
    'relu': nn.ReLU(),
    'selu': nn.SELU(),
    'silu': nn.SiLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'gelu': nn.GELU()
}


#pytorch edm model
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        kwargs = config.model_config
        
        self.lookback_len = kwargs['lookback_len']
        self.pred_len = kwargs['out_pred_len']


        self.n_edm_blocks = kwargs['n_edm_blocks']

        self.in_channels = kwargs['encoder_params']['in_channels']
        self.out_channels = kwargs['encoder_params']['in_channels']
        
        self.latent_channel_dim = kwargs['encoder_params']['latent_channel_dim']    

        self.encoder = InputEncoder(mlp_layers=kwargs['encoder_params']['mlp_layers'],
                                    lookback_len=self.lookback_len,
                                    pred_len=self.pred_len,
                                    latent_channel_dim=self.latent_channel_dim,
                                    in_channels=self.in_channels,
                                    activation_fn=activation_fn_map[kwargs['encoder_params']['activation_fn']],
                                    dropout = kwargs['encoder_params']['dropout'])
        
        
        kwargs['edm_params']['activation_fn'] = activation_fn_map[kwargs['edm_params']['activation_fn']]

        edm_blocks =  []
        for _ in range(self.n_edm_blocks):
            curr_edm = EDM(lookback_len=self.lookback_len,
                            out_pred_len=self.pred_len,
                            latent_channel_dim=self.latent_channel_dim,
                            **kwargs['edm_params'])
        
            edm_blocks.append(curr_edm)

        self.edm_blocks = nn.ModuleList(edm_blocks)
        self.gate_edm = nn.Linear(self.pred_len, 1)


    def forward(self, x, x_stamp, dec_input, batch_y_mark):    

        # B,T,D -> B,D,T
        x = x.permute(0,2,1)
        
        B,D,T = x.size()
        
        x_means = x.mean(dim=-1, keepdim=True).detach()
        x = x - x_means

        x_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x = x / x_std

        curr_lookback, mlp_edm_focal_pts, skip_focal_pts  = self.encoder(x)

        for i in range(len(self.edm_blocks)):
            curr_edm_pred, _ = self.edm_blocks[i](curr_lookback, mlp_edm_focal_pts) # B, D, final_pred_len (eg:96)
            mlp_edm_focal_pts = curr_edm_pred

        edm_pred = curr_edm_pred

        gate_prob = self.gate_edm(edm_pred).sigmoid()
        pred = (edm_pred * gate_prob) + skip_focal_pts

        pred = pred * x_std + x_means
        # B, D, final_pred_len (eg:96) -> B, final_pred_len, D
        pred = pred.permute(0,2,1)


        return pred # B, D, final_pred_len (eg:96)