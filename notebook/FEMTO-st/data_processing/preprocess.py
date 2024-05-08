import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import os
from scipy.optimize import fsolve
from functools import partial

class Process():
    def __init__(self, folders: [(str, bool, int)], window_size: int, min_signal: int, femto_path, post_process_path, is_save: bool) -> None:
        self.folders = folders
        self.life = None
        self.Y = []
        self.train = None
        self.window_size = window_size
        self.min_signal = min_signal
        self.is_save = is_save
        self.FEMTO = femto_path
        self.POST_PROCESS= post_process_path
        
    def _equation(self, tau, a, convergence):
        return 1 + np.exp(a) - np.exp((convergence * tau) + a)
    
    def HI(self, t, a , tau):
        return 1 + np.exp(a) - np.exp((t * tau) + a)
    
    def loop_folder(self):
        for (folder, is_train, convergence) in self.folders:
            self.Y = []
            feature1s = []
            feature2s = []
            feature3s = []
            print(folder, is_train)
            try:
                self.train = is_train
                accs = os.listdir(self.FEMTO+folder)
                accs.sort()
                accs = [acc for acc in accs if acc.startswith('acc')]
                self.life = len(accs)
                for stamp, acc in enumerate(accs):
                    feature1, feature2, feature3 = self._get_feature(f'{self.FEMTO}/{folder}/{acc}')
                    feature1s.append(feature1)
                    feature2s.append(feature2)
                    feature3s.append(feature3)
                feature1s = self._slide_x_window(feature1s)
                feature2s = self._slide_x_window(feature2s)
                feature3s = self._slide_x_window(feature3s)
                y_HI = self._reconstruct_HI(convergence)
                if self.is_save:
                    self._save_x_data(folder, feature1s, '2560')
                    self._save_x_data(folder, feature2s, '1280')
                    self._save_x_data(folder, feature3s, '640')
                    self.Y = self._slide_y_window(y_HI)
                    self._save_y_data(folder)
            except Exception as e:
                print(e)
                print(folder)
                
    def _save_x_data(self, folder: str, features, signal_size: int):
        np.save(f'{self.POST_PROCESS}{folder}/{folder}_X_17_{signal_size}', features)

    def _save_y_data(self, folder):
        np.save(f'{self.POST_PROCESS}{folder}/{folder}_Y', self.Y)
        
    def _reconstruct_HI(self, convergence: int):
        initial_guess = 0
        a = 1
        result = fsolve(self._equation, initial_guess, args=(a, convergence))
        tau = result[0]
        print(f"The solution for τ is: {tau}")
        partial_HI = partial(self.HI, a=a, tau=tau)
        rul = [i for i in range(self.life)]
        hi_y = list(map(partial_HI, rul))
        min_value = min(hi_y)
        max_value = max(hi_y)
        normalized_values = [(x - min_value) / (max_value - min_value) for x in hi_y]
        return normalized_values

    def _get_feature(self, acc: str):
        x = pd.read_csv(acc, header=None, sep=',', usecols=[4])
        feature1 = self._extract_feature(x, self.min_signal * 4)
        tmp1 = self._extract_feature(x[:2 * self.min_signal], self.min_signal * 2)
        tmp2 = self._extract_feature(x[2 * self.min_signal:], self.min_signal * 2)
        feature2 = tmp1 + tmp2
        tmp1 = self._extract_feature(x[:self.min_signal], self.min_signal)
        tmp2 = self._extract_feature(x[self.min_signal:2 * self.min_signal], self.min_signal)
        tmp3 = self._extract_feature(x[2 * self.min_signal:3 * self.min_signal], self.min_signal)
        tmp4 = self._extract_feature(x[3 * self.min_signal:4 * self.min_signal], self.min_signal)
        feature3 = tmp1 + tmp2 + tmp3 + tmp4
        return feature1, feature2, feature3
    
    def _extract_feature(self, x: pd.DataFrame, LEN: int):
        # time zone
        x_abs = x.abs()
        x_avg = x.mean()
        x_sub_mean = x.sub(x_avg, axis=1)
        mean_square_sum = (x_sub_mean ** 2).sum()
        p1 = x.max()
        p2 = x.min()
        p3 = x_abs.max()
        p4 = p1 - p2
        p5 = x_abs.sum() / LEN
        p6 = (x_abs.sum() ** 0.5 / LEN) * 2
        p7 = mean_square_sum / (LEN -1)
        p8 = (mean_square_sum / LEN) ** 0.5
        p9 = ((x ** 2).sum() / LEN) ** 0.5
        # p10 = (x_sub_mean ** 4).sum() / ((LEN - 1) * ((mean_square_sum / LEN)  ** 2))
        p11 = (LEN * p9) / x_abs.sum()
        p12 = p9 / p5
        p13 = p3 / p9
        p14 = p3 / p5
        p15 = p3 / p6
        p16 = p3 / (p9 ** 2)
        
        # frequency zone
        fft_result = np.fft.fft(x.to_numpy(), axis=0)
        N = len(fft_result)
        amplitudes = np.abs(fft_result)
        p17 = np.sum(amplitudes) / N
        return [p1.iloc[0], p2.iloc[0], p3.iloc[0], p4.iloc[0], p5.iloc[0], p6.iloc[0], 
                p7.iloc[0], p8.iloc[0], p9.iloc[0], p11.iloc[0], p12.iloc[0], 
                p13.iloc[0], p14.iloc[0], p15.iloc[0], p16.iloc[0], p17]
    
    def _slide_y_window(self, y_hi):
        y_windows = []
        for i in range(self.life - self.window_size):
            y_window = np.array(y_hi)[i + 40]
            y_windows.append(y_window)
        return np.array(y_windows)

    def _slide_x_window(self, features):
        feature_windows = []
        for i in range(self.life - self.window_size):
            feature_window = np.array(features)[i:i + self.window_size, :]
            feature_windows.append(feature_window)
        return np.array(feature_windows)
    
