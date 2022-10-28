import csv
import os
import numpy as np


class Recorder:
    def __init__(self, args, dir_name):
        self.metric_dir = os.path.join(args.result_dir, dir_name)
        os.makedirs(self.metric_dir, exist_ok=True)
        self.f = open(os.path.join(self.metric_dir, f'{args.exp_name}.csv'), 'w', newline='')
        self.writer = csv.writer(self.f)
        self.metric_names = ['MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE']
        self.columns = ['Label'] + self.metric_names
        self.metrics = {'MSE': [], 'MAE': [], 'RMSE': [], 'MAPE': [], 'MSPE': []}
        self.writer.writerow(self.columns)
    
    def writerow(self, iter, metric_dict):
        self.writer.writerow([str(iter)] + [str(metric_dict[metric_name]) for metric_name in self.metric_names])
        for key, value in metric_dict.items():
            self.metrics[key].append(value)
    
    def write_statistics(self):
        self.writer.writerow(['Mean'] + [str(np.mean(self.metrics[metric_name])) for metric_name in self.metric_names])
        self.writer.writerow(['SD'] + [str(np.std(self.metrics[metric_name])) for metric_name in self.metric_names])
    
    def close(self):
        self.f.close()
