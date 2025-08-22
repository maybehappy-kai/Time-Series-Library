import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import yaml

#plt.switch_backend('agg')

class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }  # different predict length
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == "custom":
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
            new_lr = max(curr_lr * args.opt['reduce_lr_factor'],  args.opt['min_lr'])
            if curr_lr != new_lr:
                print(f"Reducing learning rate to {new_lr:.6f}")
            param_group['lr'] = new_lr
        return
        
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def tdt_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # Calculate TDT using first-order differencing
    dt_true = y_true[:, :, 1:] - y_true[:, :, :-1]
    dt_pred = y_pred[:, :, 1:] - y_pred[:, :, :-1]

    # TDT Values Prediction Loss (LD)
    LD = torch.mean(torch.abs(dt_pred - dt_true))

    # Adaptive Weight (ρ)
    sign_diff = torch.sign(dt_true) != torch.sign(dt_pred)
    rho = torch.mean(sign_diff.float())

    return rho, LD


def load_and_merge_cfg_args(args):
    if os.path.exists(args.model_config):
        with open(args.model_config, "r") as fd:
            cfg = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        return args

    if args.seed == -1 or args.seed is None or args.seed == "":
        args.seed = cfg['rand_seed']
    
    print(f"Setting seed to {args.seed}")  

    args.condor_job  = args.condor_job.lower() == "true" 

    loss_type = args.loss
    args.loss_type = loss_type.lower().split("_")[0]
    args.tdt_loss = "tdt" in loss_type

    csv_name = args.data_path
    if csv_name in ['ETTh1.csv', 'ETTh2.csv','ETTm1.csv','ETTm2.csv', 'national_illness.csv']:
        in_channels = 7
    elif csv_name in ['weather.csv']:
        in_channels = 21
    elif csv_name in ['traffic.csv']:
        in_channels = 862
    elif csv_name in ['exchange_rate.csv']:
        in_channels = 8
    elif csv_name in ['electricity.csv']:
        in_channels = 321
    elif 'lorenz' in csv_name or 'rossler' in csv_name:
        in_channels = 3
    elif "solar" in csv_name:
        in_channels = 137
    elif "m4" in args.root_path:
        in_channels = 1
    else:
        if args.in_channels == -1:
            raise ValueError("Unknown dataset")
        in_channels = args.in_channels

    cfg['model']['encoder_params']['activation_fn'] = args.activation_fn
    cfg['model']['edm_params']['activation_fn'] = args.activation_fn

    cfg['model']['encoder_params']['in_channels'] = in_channels
    
    if args.n_mlp_layers != -1:
        cfg['model']['encoder_params']['mlp_layers'] = args.n_mlp_layers
    
    if args.n_edm_blocks != -1:
        cfg['model']['n_edm_blocks'] = args.n_edm_blocks

    cfg['model']['encoder_params']['dropout'] = args.mlp_dropout
    cfg['model']['edm_params']['dropout'] = args.edm_dropout

    if args.dist_projection_dim != -1:
        cfg['model']['edm_params']['dist_projection_dim'] = args.dist_projection_dim
    
    args.add_pe = args.add_pe.lower() == "true"
    if args.add_pe:
        cfg['model']['encoder_params']['add_pe'] = args.add_pe


    if args.latent_channel_dim != -1:
        cfg['model']['encoder_params']['latent_channel_dim'] = args.latent_channel_dim
    else:
        cfg['model']['encoder_params']['latent_channel_dim'] = in_channels


    if args.delay != -1:
        cfg['model']['edm_params']['delay'] = args.delay
    
    if args.time_delay_stride != -1:
        cfg['model']['edm_params']['time_delay_stride'] = args.time_delay_stride

    if args.n_proj_layers != -1:
        cfg['model']['edm_params']['n_proj_layers'] = args.n_proj_layers


    if args.data == 'm4':
        args.pred_len = M4Meta.horizons_map[args.seasonal_patterns]  # Up to M4 config
        args.seq_len = 2 * args.pred_len  # input_len = 2*pred_len
        args.label_len = args.pred_len

    cfg['model']['lookback_len'] = args.seq_len
    cfg['model']['out_pred_len'] = args.pred_len

    if args.theta != -1:
        cfg['model']['edm_params']['theta'] = args.theta
    

    # #training parameters
    if args.learning_rate == -1:
        args.learning_rate = cfg['opt']['learning_rate']

    if args.min_lr != -1:
        cfg['opt']['min_lr'] = args.min_lr
    
    if args.reduce_lr_factor != -1:
        cfg['opt']['reduce_lr_factor'] = args.reduce_lr_factor

    args.clip_grad_norm = cfg['opt']['clip_grad_norm']

    if args.train_epochs == -1:
        args.train_epochs = cfg['opt']['epochs']

    if args.patience == -1:
        args.patience = cfg['opt']['early_stopping_patience']

    if args.lradj == "":
        args.lradj = cfg['opt']['schedule_type']
    
    args.opt = cfg['opt']

    args.model_config = cfg['model']

    return args