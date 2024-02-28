import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DAST_utils import *
import sys
DEBUG = False


class Sensors_EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.1):
        super(Sensors_EncoderLayer, self).__init__()
        self.attn = Sensor_MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if DEBUG == True:
            print('Sensor encoder input: ', x.shape)
        a = self.attn(x)
        if DEBUG == True:
            print('Sensor encoder attn: ', a.size())
        x = self.norm1(x + a)
        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + a)
        return x


class Time_step_EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.1):
        super(Time_step_EncoderLayer, self).__init__()
        self.attn = TimeStepMultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)

    def forward(self, x):
        if DEBUG == True:
            print('Time encoder input: ', x.size())
        a = self.attn(x)
        if DEBUG == True:
            print('Time encoder attn: ', a.size())
        x = self.norm1(x + a)
        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + a)
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)

    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)
        a = self.attn2(x, kv=enc)
        x = self.norm2(a + x)
        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm3(x + a)
        return x


class DAST(torch.nn.Module):
    def __init__(self, dim_val_s, dim_attn_s, dim_val_t, dim_attn_t, dim_val, dim_attn, time_step,
                 feature_len, dec_seq_len, out_seq_len, n_decoder_layers=1, n_encoder_layers=1,
                 n_heads=1, debug=False, dropout=0.1):

        super(DAST, self).__init__()
        self.dec_seq_len = dec_seq_len
        self.dropout = nn.Dropout(dropout)
        self.debug = debug
        # Initiate Sensors encoder
        self.sensor_encoder1 = nn.ModuleList()  # []
        self.sensor_encoder2 = nn.ModuleList()
        self.sensor_encoder3 = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.sensor_encoder1.append(
                Sensors_EncoderLayer(dim_val_s, dim_attn_s, n_heads))
            self.sensor_encoder2.append(
                Sensors_EncoderLayer(dim_val_s, dim_attn_s, n_heads))
            self.sensor_encoder3.append(
                Sensors_EncoderLayer(dim_val_s, dim_attn_s, n_heads))
        # Initiate Time_step encoder
        self.time_encoder = nn.ModuleList()   # []
        for i in range(n_encoder_layers):
            self.time_encoder.append(Time_step_EncoderLayer(
                dim_val_t, dim_attn_t, n_heads))

        # Initiate Decoder
        self.decoder1 = nn.ModuleList()
        self.decoder2 = nn.ModuleList()
        self.decoder3 = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decoder1.append(DecoderLayer(dim_val, dim_attn, n_heads))
            self.decoder2.append(DecoderLayer(dim_val, dim_attn, n_heads))
            self.decoder3.append(DecoderLayer(dim_val, dim_attn, n_heads))

        self.pos_s = PositionalEncoding(dim_val_s)
        self.pos_t = PositionalEncoding(dim_val_t)
        self.timestep_enc_input_fc = nn.Linear(feature_len, dim_val_t)
        self.sensor_enc_input_fc = nn.Linear(time_step, dim_val_s)
        self.dec_input_fc1 = nn.Linear(feature_len, dim_val)
        self.dec_input_fc2 = nn.Linear(feature_len * 2, dim_val)
        self.dec_input_fc3 = nn.Linear(feature_len * 4, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val * 3, out_seq_len)
        self.norm1 = nn.LayerNorm(dim_val)
        self.feature_len = feature_len

    def forward(self, x):
        sensor_x = x.transpose(1, 2)
        feature1 = x[:, :, :self.feature_len].transpose(1, 2)
        feature2 = x[:, :, self.feature_len: 3 *
                     self.feature_len].transpose(1, 2)
        feature3 = x[:, :, 3 * self.feature_len: 7 *
                     self.feature_len].transpose(1, 2)
        x = x[:, :, :self.feature_len]
        # input embedding and positional encoding
        if DEBUG == True:
            print('x shape:', x.shape)
            print('input X: ', x.size())
            print('sensor X: ', sensor_x.size())
            print(feature1.shape, feature2.shape, feature3.shape, sep='\n')
            print(self.sensor_enc_input_fc(feature1).shape)
            print(self.sensor_enc_input_fc(feature2).shape)
            print(self.sensor_enc_input_fc(feature3).shape)
        
        e1 = self.sensor_encoder1[0](self.sensor_enc_input_fc(
            feature1))  # ((batch_size,sensor,dim_val_s))
        e2 = self.sensor_encoder1[0](self.sensor_enc_input_fc(
            feature2))
        e3 = self.sensor_encoder1[0](self.sensor_enc_input_fc(
            feature3))
        if DEBUG == True:
            print('sensor encoder X: ', e1.shape, e2.shape, e3.shape, )
            print(self.timestep_enc_input_fc(x).shape)
        # ((batch_size,timestep,dim_val_t))
        o = self.time_encoder[0](self.pos_t(self.timestep_enc_input_fc(x)))
        if DEBUG == True:
            print('time encoder X: ', o.size())
        # sensors encoder
        for sensor_enc in self.sensor_encoder1[1:]:
            e1 = sensor_enc(e1)
        for sensor_enc in self.sensor_encoder2[1:]:
            e2 = sensor_enc(e2)
        for sensor_enc in self.sensor_encoder3[1:]:
            e3 = sensor_enc(e3)
        # time step encoder
        for time_enc in self.time_encoder[1:]:
            o = time_enc(o)

        # feature fusion
        p1 = torch.cat((e1, o), dim=1)  # ((batch_size,timestep+sensor,dim_val))
        p1 = self.norm1(p1)

        p2 = torch.cat((e2, o), dim=1)
        p2 = self.norm1(p2)
        
        p3 = torch.cat((e3, o), dim=1)
        p3 = self.norm1(p3)
        # decoder receive the output of feature fusion layer.
        if DEBUG == True:
            print('decoder X: ', p1.size())
            print('f1 decode no transpose: ', feature1[:, -self.dec_seq_len:].shape)
            print('f1 decode: ', feature1.transpose(1, 2)[:, -self.dec_seq_len:].shape)
        d1 = self.decoder1[0](self.dec_input_fc1(feature1.transpose(1, 2)[:, -self.dec_seq_len:]), p1)
        d2 = self.decoder2[0](self.dec_input_fc2(feature2.transpose(1, 2)[:, -self.dec_seq_len:]), p2)
        d3 = self.decoder3[0](self.dec_input_fc3(feature3.transpose(1, 2)[:, -self.dec_seq_len:]), p3)
        if DEBUG == True:
            print('embedding decoder X: ', d1.size())
        # output the RUL
        d = torch.cat((d1, d2, d3), dim=2)
        x = self.out_fc(F.elu(d.flatten(start_dim=1)))
        if DEBUG == True:
            print('x: ', x.shape)
            print('d1+d2+d3: ', d.shape)
            print('output X: ', x.size())
        return x
