import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DAST_utils import *
import sys
DEBUG = True


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


class Sensors_EncoderLayer2(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.1):
        super(Sensors_EncoderLayer, self).__init__()
        dim_val = dim_val * 2
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


class Sensors_EncoderLayer3(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.1):
        super(Sensors_EncoderLayer, self).__init__()
        dim_val = dim_val * 3
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
                 input_size, dec_seq_len, out_seq_len, n_decoder_layers=1, n_encoder_layers=1,
                 n_heads=1, feature_len=16, debug=False, dropout=0.1):

        super(DAST, self).__init__()
        self.dec_seq_len = dec_seq_len
        self.dropout = nn.Dropout(dropout)
        self.debug = debug
        # Initiate Sensors encoder
        self.sensor_encoder = nn.ModuleList()  # []
        for i in range(n_encoder_layers):
            self.sensor_encoder.append(
                Sensors_EncoderLayer(dim_val_s, dim_attn_s, n_heads))
        # Initiate Time_step encoder
        self.time_encoder = nn.ModuleList()   # []
        for i in range(n_encoder_layers):
            self.time_encoder.append(Time_step_EncoderLayer(
                dim_val_t, dim_attn_t, n_heads))

        # Initiate Decoder
        self.decoder = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decoder.append(DecoderLayer(dim_val, dim_attn, n_heads))

        self.pos_s = PositionalEncoding(dim_val_s)
        self.pos_t = PositionalEncoding(dim_val_t)
        self.timestep_enc_input_fc = nn.Linear(input_size, dim_val_t)
        self.sensor_enc_input_fc = nn.Linear(time_step, dim_val_s)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)
        self.norm1 = nn.LayerNorm(dim_val)
        self.feature_len = feature_len

    def forward(self, x):
        # input embedding and positional encoding
        x = x[:, :, :self.feature_len]
        if self.debug == True:
            print('input X: ', x.size())
        sensor_x = x.transpose(1, 2)
        if self.debug == True:
            print('sensor X: ', sensor_x.size())
        print(self.sensor_enc_input_fc(sensor_x).shape)
        e = self.sensor_encoder[0](self.sensor_enc_input_fc(
            sensor_x))  # ((batch_size,sensor,dim_val_s))
        if self.debug == True:
            print('sensor encoder X: ', e.size())
        print(self.timestep_enc_input_fc(x).shape)
        # ((batch_size,timestep,dim_val_t))
        o = self.time_encoder[0](self.pos_t(self.timestep_enc_input_fc(x)))
        if self.debug == True:
            print('time encoder X: ', o.size())
        # sensors encoder
        for sensor_enc in self.sensor_encoder[1:]:
            e = sensor_enc(e)

        # time step encoder
        for time_enc in self.time_encoder[1:]:
            o = time_enc(o)

        # feature fusion
        p = torch.cat((e, o), dim=1)  # ((batch_size,timestep+sensor,dim_val))
        p = self.norm1(p)

        # decoder receive the output of feature fusion layer.
        if self.debug == True:
            print('decoder X: ', p.size())
        print('decode left input: ', x[:, -self.dec_seq_len:].shape)
        print('decode left input embed: ', self.dec_input_fc(x[:, -self.dec_seq_len:]).shape)
        d = self.decoder[0](self.dec_input_fc(x[:, -self.dec_seq_len:]), p)
        if self.debug == True:
            print('embedding decoder X: ', d.size())
        # output the RUL
        # x = self.out_fc(d.flatten(start_dim=1))
        x = self.out_fc(F.elu(d.flatten(start_dim=1)))
        if self.debug == True:
            print('output X: ', x.size())
        return x
