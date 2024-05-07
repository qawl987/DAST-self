import sys
sys.path.append('/home/kai/DAST/network')
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from torch.autograd import Variable
from DAST_utils import *
from DAST_Network import *
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import List
from enums import NormType, TrainType
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class DASTTF(nn.Module):
    def __init__(self, HP, pretrain_model_path, pretrain_model_name):
        super(DASTTF, self).__init__()
        self.HP = HP
        self.pretrained_model = DAST(self.HP['dim_val_s'], self.HP['dim_attn_s'], self.HP['dim_val_t'], self.HP['dim_attn_t'], self.HP['dim_val'], self.HP['dim_attn'], self.HP['time_step'], self.HP['feature_len'], self.HP['dec_seq_len'], self.HP['output_sequence_length'], self.HP['n_decoder_layers'], self.HP['n_encoder_layers'], self.HP['n_heads'], self.HP['debug'])
        self.pretrained_model.load_state_dict(torch.load(f'{pretrain_model_path}/{pretrain_model_name}.pt'))
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
            
        for param in self.pretrained_model.out_fc.parameters():
            param.requires_grad = True
            
        for param in self.pretrained_model.sensor_enc_input_fc.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        # Forward pass through the pretrained model
        fine_tuned_output = self.pretrained_model(x)
        return fine_tuned_output
    
class DASTModel():
    def __init__(self, train_datasets: List[str], test_dataset: List[str], data_path: str, norm_type: NormType, hyper_parameters: dict, model_save_path: str, model_save_name: str, is_save_model: bool, train_type: TrainType) -> None:
        self.TRAIN_DATASETS = train_datasets
        self.TEST_DATASETS = test_dataset
        self.DATA_PATH = data_path
        self.MODEL_SAVE_PATH = model_save_path
        self.MODEL_SAVE_NAME = model_save_name
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        self.HP = hyper_parameters
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.best_predict = []
        self.last_predict_y = None
        self.train_loss_list = []
        self.test_loss_list = []
        self.best_mse_loss = 10000.0
        self.best_rmse_loss = None
        self.best_train_loss = 10000.0
        self.norm_type = norm_type
        self.best_model_params = None
        self.IS_SAVE_MODEL = is_save_model
        self.TRAIN_TYPE = train_type
        self.pretrain_model = None

    @staticmethod
    def RMSE(target, pred):
        square_error = (target - pred) ** 2
        mse =  (torch.sum(square_error)) / len(target)
        rmse = mse ** 0.5
        return rmse
    
    @staticmethod
    def MAE(target, pred):
        absolute_error = np.abs(target - pred)
        return torch.sum(absolute_error) / len(target)
    
    def _load_x_y(self, folder: str):
        y_tmp = np.load(f'{self.DATA_PATH}/{folder}/{folder}_Y.npy')
        feature1 = self._norm(np.load(f'{self.DATA_PATH}/{folder}/{folder}_X_17_2560.npy'))
        feature2 = self._norm(np.load(f'{self.DATA_PATH}/{folder}/{folder}_X_17_1280.npy'))
        feature3 = self._norm(np.load(f'{self.DATA_PATH}/{folder}/{folder}_X_17_640.npy'))
        X_train = np.concatenate((feature1, feature2, feature3), axis=2)
        return X_train, np.reshape(y_tmp, ((len(y_tmp), -1)))
    
    def _norm(self, array):
        if self.norm_type == NormType.NO_NORM:
            return array
        elif self.norm_type == NormType.BATCH_NORM:
            feat_len = array.shape[2]
            x = array.reshape(-1, feat_len)
            min_vals = np.min(x, axis=1, keepdims=True)
            max_vals = np.max(x, axis=1, keepdims=True)
            normalized_array = (x - min_vals) / (max_vals - min_vals)
            original_shape_array = normalized_array.reshape(-1, 40, feat_len)
            return original_shape_array
        elif self.norm_type == NormType.LAYER_NORM:
            min_list = []
            max_list = []
            feat_len = array.shape[2]
            x = array.reshape(-1, feat_len)
            for i in range(feat_len):
                col = x[:, i]
                mi = np.min(col)
                ma = np.max(col)
                min_list.append(mi)
                max_list.append(ma)
            min_array = np.array(min_list)
            max_array = np.array(max_list)
            norm_array = (x - min_array) / (max_array - min_array)
            original_shape_array = norm_array.reshape(-1, 40, feat_len)
            return original_shape_array
        
    def _concate(self):
        self.X_train = np.concatenate(self.X_train, axis=0)
        self.Y_train = np.concatenate(self.Y_train, axis=0)
        self.X_test = np.concatenate(self.X_test, axis=0)
        self.Y_test = np.concatenate(self.Y_test, axis=0)

    def _load_np(self,):
        # train
        for folder in self.TRAIN_DATASETS:
            X_train, Y_train = self._load_x_y(folder)
            # print(X_train.shape)
            # print(X_train[0][0])
            self.X_train.append(X_train)
            self.Y_train.append(Y_train)
        # test
        for folder in self.TEST_DATASETS:
            X_test, Y_test = self._load_x_y(folder)
            self.X_test.append(X_test)
            self.Y_test.append(Y_test)
        
    def _loop_feature(self, X, selected_indices):
        extracted_values_list = []
        for i in range(7):
            for num in selected_indices:
                extracted_values = X[:, :, num + 16 * i]
                extracted_values_list.append(extracted_values)
        result_array = np.stack(extracted_values_list, axis=-1)
        return result_array
    
    def _select_feature(self, selected_indices):
        for i in range(len(self.X_train)):
            self.X_train[i] = self._loop_feature(self.X_train[i], selected_indices)
        for i in range(len(self.X_test)):
            self.X_test[i] = self._loop_feature(self.X_test[i], selected_indices)
        
    def _tensorizing(self):
        self.X_train = Variable(torch.Tensor(self.X_train).float())
        self.Y_train = Variable(torch.Tensor(self.Y_train).float())
        self.X_test = Variable(torch.Tensor(self.X_test).float())
        self.Y_test = Variable(torch.Tensor(self.Y_test).float())
        
    def _get_dataloader(self):
        train_dataset = TensorDataset(self.X_train, self.Y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.HP['batch_size'], shuffle=False)
        test_dataset = TensorDataset(self.X_test, self.Y_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.HP['batch_size'], shuffle=False)
        return train_loader, test_loader
    
    def _get_model(self):
        if self.TRAIN_TYPE == TrainType.DL:
            model = DAST(self.HP['dim_val_s'], self.HP['dim_attn_s'], self.HP['dim_val_t'], self.HP['dim_attn_t'], self.HP['dim_val'], self.HP['dim_attn'], self.HP['time_step'], self.HP['feature_len'], self.HP['dec_seq_len'], self.HP['output_sequence_length'], self.HP['n_decoder_layers'], self.HP['n_encoder_layers'], self.HP['n_heads'], self.HP['debug'])
        elif self.TRAIN_TYPE == TrainType.TL:
            model = self.pretrain_model
        # model = DAST(self.HP['dim_val_s'], self.HP['dim_attn_s'], self.HP['dim_val_t'], self.HP['dim_attn_t'], self.HP['dim_val'], self.HP['dim_attn'], self.HP['time_step'], self.HP['feature_len'], self.HP['dec_seq_len'], self.HP['output_sequence_length'], self.HP['n_decoder_layers'], self.HP['n_encoder_layers'], self.HP['n_heads'], self.HP['debug'])
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.HP['lr'])
        criterion = nn.MSELoss()
        return model, optimizer, criterion

    def set_model(self, pretrain_model):
        self.pretrain_model = pretrain_model
        
    def train(self, model: DAST, optimizer: torch.optim.Optimizer, criterion, train_loader: DataLoader, epoch: int):
        model.train()
        tmp_loss_list = []
        loop = tqdm(train_loader, leave=True)
        for _, (X, Y) in enumerate(loop):
            batch_X = X.to(self.device)
            batch_Y = Y.to(self.device)
            out = model(batch_X)
            loss = criterion(out, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp_loss_list.append(loss.item())
        loss_eopch = np.mean(np.array(tmp_loss_list))
        self.train_loss_list.append(loss_eopch)
        if (loss_eopch.item() < self.best_train_loss):
            self.best_train_loss = loss_eopch.item()
        print('epoch = ',epoch,
                'train_loss = ',loss_eopch.item())

    def eval(self, model: DAST, test_loader: DataLoader, criterion, epoch: int):
        model.eval()
        prediction_list = []
        with torch.no_grad():
            for _ ,(batch_x, _) in enumerate(test_loader):
                batch_X = batch_x.to(self.device)
                prediction = model(batch_X)
                prediction_list.append(prediction)

        out_batch_pre = torch.cat(prediction_list).detach().cpu()
        rmse_loss = self.RMSE(self.Y_test, out_batch_pre, )
        mae_loss = self.MAE(self.Y_test, out_batch_pre, )
        test_loss = criterion(out_batch_pre, self.Y_test)
        self.test_loss_list.append(test_loss)
        if (test_loss.item() < self.best_mse_loss):
            self.best_mse_loss = test_loss.item()
            self.best_rmse_loss = rmse_loss.item()
            self.best_predict = np.reshape(out_batch_pre, (-1)).tolist()
            if self.IS_SAVE_MODEL:
                torch.save(model.state_dict(), f'{self.MODEL_SAVE_PATH}/{self.MODEL_SAVE_NAME}.pt')
        print('rmse_loss = ', rmse_loss.item(),
                'mae_loss = ', mae_loss.item(),
                'mse_loss = ', test_loss.item())
        if epoch == self.HP['epochs'] - 1:
            self.last_predict_y = out_batch_pre

    def main(self, selected_indices):
        self._load_np()
        self._select_feature(selected_indices)
        self._concate()
        self._tensorizing()
        model, optimizer, criterion = self._get_model()
        train_loader, test_loader = self._get_dataloader()
        times = 0
        for epoch in range(self.HP['epochs']):
            start = time.time()
            self.train(model, optimizer, criterion, train_loader, epoch)
            end = time.time()
            times += end - start
            self.eval(model, test_loader, criterion, epoch)
        print(f"train time: {times/100:.7f}, s/epoch")
        print(f"embed1: {self.HP['dim_val_s']}, embed2: {self.HP['dim_attn_s']}, lr: {self.HP['lr']}, dec_seq_len: {self.HP['dec_seq_len']}")
        print(f"{self.best_train_loss:.7f}")
        print(f"{self.best_mse_loss:.7f}")
        print(f"{self.best_rmse_loss:.7f}")
        
    def plt_loss_list(self):
        plt.title("Train Test Loss Curve")
        plt.plot(self.train_loss_list, label='Train loss', marker='o', markersize=1)
        plt.plot(self.test_loss_list, label='Test loss', marker='s', markersize=1)
        plt.xlabel('Epoch')
        plt.ylabel('MSE loss')
        plt.legend()
        plt.show()
        
    def plt_best_predict(self):
        y = self.Y_test.detach().cpu().numpy()
        y = np.reshape(y, -1)
        y = y.tolist()
        plt.title("Test Best Predict VS. Truth")
        plt.plot(self.best_predict, label='Pred', marker='o', markersize=1)
        plt.plot(y, label='Y', marker='s', markersize=1)
        plt.xlabel('Time Step')
        plt.ylabel('RUL')
        plt.legend()
        plt.show()
        
    def plt_last_predict_vs_y(self):
        last_predict_y = np.reshape(self.last_predict_y, (-1))
        last_predict_y = last_predict_y.tolist()
        y = self.Y_test.detach().cpu().numpy()
        y = np.reshape(y, -1)
        y = y.tolist()
        plt.title("Test Last Predict VS. Truth")
        plt.plot(last_predict_y, label='Pred', marker='o', markersize=1)
        plt.plot(y, label='Y', marker='s', markersize=1)
        plt.xlabel('Time Step')
        plt.ylabel('RUL')
        plt.legend()
        plt.show()