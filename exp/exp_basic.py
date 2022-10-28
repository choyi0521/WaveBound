import os
import torch
import numpy as np


class ExpBasic:
    def __init__(self, args, exp_num, logger):
        self.args = args
        self.exp_num = exp_num
        self.logger = logger
        self.device = self._acquire_device()

        self.exp_dir = os.path.join(self.args.checkpoint_dir, self.args.exp_name, str(exp_num))
        os.makedirs(self.exp_dir, exist_ok=True)

        self.best_checkpoint_path = os.path.join(self.exp_dir, 'best_checkpoint.pth')

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            self.logger.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            self.logger.info('Use CPU')
        return device
