from data_provider.data_factory import data_provider
from exp.exp_basic import ExpBasic
from models import Informer, Autoformer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, reset_batchnorm_statistics, add_gaussian_noise
from utils.metrics import get_metrics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from utils.ema import EMAUpdater

warnings.filterwarnings('ignore')


class ExpMain(ExpBasic):
    def __init__(self, args, exp_num, logger):
        super().__init__(args, exp_num, logger)
        self.source_model = self._build_model().to(self.device)
        
        self.use_ema = self.args.use_ema
        if self.use_ema:
            self.target_model = self._build_model().to(self.device)
            self.ema_updater = EMAUpdater(self.target_model, self.source_model, self.args.moving_average_decay, self.args.start_iter)
            self.eval_model = self.source_model if self.args.ema_eval_model == 'source' else self.target_model
        else:
            self.eval_model = self.source_model
        
        self.f_dim = -1 if self.args.features == 'MS' else 0
        self.mse_loss = nn.MSELoss()
        
        self.train_loader = self._get_dataloader(flag='train')
        self.valid_loader = self._get_dataloader(flag='valid')
        self.test_loader = self._get_dataloader(flag='test')

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_dataloader(self, flag):
        return data_provider(self.args, flag, self.logger)[1]

    def _select_optimizer(self):
        if self.args.optimizer == 'sgd':
            model_optim = optim.SGD(self.source_model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adam':
            model_optim = optim.Adam(self.source_model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adamw':
            model_optim = optim.AdamW(self.source_model.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError
        return model_optim
    
    def get_decoder_input(self, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
        return dec_inp

    def get_output(self, model, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y=None):
        if self.args.output_attention:
            output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)[0]
        else:
            output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
        return output
    
    def compute_loss(self, target_output, source_output, batch_y, loss_type):
        batch_y = batch_y[:, -self.args.pred_len:, self.f_dim:]

        if loss_type == 'BDFMSE':
            target_output = target_output[:, -self.args.pred_len:, self.f_dim:]
            source_output = source_output[:, -self.args.pred_len:, self.f_dim:]
            loss = torch.abs(((source_output - batch_y)**2).mean(0) - ((target_output - batch_y)**2).mean(0) + self.args.epsilon).mean()
        elif loss_type == 'MSE':
            source_output = source_output[:, -self.args.pred_len:, self.f_dim:]
            loss = self.mse_loss(source_output, batch_y)
        else:
            raise NotImplementedError
        
        return loss

    def train(self):
        time_now = time.time()
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(logger=self.logger, patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        iter_count = 0
        for epoch in range(self.args.train_epochs):
            train_loss = []

            if self.use_ema:
                self.target_model.train()
            self.source_model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self.get_decoder_input(batch_y)

                # encoder - decoder
                if self.use_ema and iter_count >= self.args.start_iter:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            target_output = self.get_output(self.target_model, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                            source_output = self.get_output(self.source_model, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                            loss = self.compute_loss(target_output, source_output, batch_y, self.args.ema_loss)
                    else:
                        target_output = self.get_output(self.target_model, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                        source_output = self.get_output(self.source_model, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                        loss = self.compute_loss(target_output, source_output, batch_y, self.args.ema_loss)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            source_output = self.get_output(self.source_model, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                            loss = self.compute_loss(None, source_output, batch_y, self.args.loss)
                    else:
                        source_output = self.get_output(self.source_model, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                        loss = self.compute_loss(None, source_output, batch_y, self.args.loss)
                
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    self.logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / 100
                    left_time = speed * (train_steps - i)
                    self.logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    time_now = time.time()
                
                model_optim.zero_grad()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                if self.use_ema:
                    self.ema_updater.update(iter_count)
                iter_count += 1

            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            if self.args.use_ema:
                self.apply_standing_statistics()
            vali_loss = self.valid(self.valid_loader)
            test_loss = self.valid(self.test_loader)

            self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.eval_model, self.best_checkpoint_path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args, self.logger)

    def valid(self, data_loader):
        total_loss = []
        self.eval_model.eval()
        n_samples = len(data_loader.dataset)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self.get_decoder_input(batch_y)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        eval_output = self.get_output(self.eval_model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        loss = self.compute_loss(None, eval_output, batch_y, 'MSE')
                else:
                    eval_output = self.get_output(self.eval_model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    loss = self.compute_loss(None, eval_output, batch_y, 'MSE')

                total_loss.append(loss.item() * batch_x.shape[0] / n_samples)
        total_loss = np.sum(total_loss)
        return total_loss

    def test(self, phase='test', load_best_checkpoint=True):
        if load_best_checkpoint:
            self.logger.info('loading model')
            self.eval_model.load_state_dict(torch.load(self.best_checkpoint_path))

        preds = []
        trues = []

        if phase == 'train':
            data_loader = self.train_loader
        elif phase == 'valid':
            data_loader = self.valid_loader
        elif phase == 'test':
            data_loader = self.test_loader
        else:
            raise NotImplementedError

        self.eval_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self.get_decoder_input(batch_y)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        eval_output = self.get_output(self.eval_model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    eval_output = self.get_output(self.eval_model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                pred = eval_output[:, -self.args.pred_len:, self.f_dim:].detach().cpu().numpy()
                true = batch_y[:, -self.args.pred_len:, self.f_dim:].detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        self.logger.info(f'Phase: {phase} - pred shape: {preds.shape}')

        # result save
        mse, mae, rmse, mape, mspe = get_metrics(preds, trues)
        self.logger.info(f'Phase: {phase} - MSE: {mse}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}, MSPE: {mspe}')
        # with open(os.path.join(self.exp_dir, "result.txt"), "a") as f:
        #     f.write('mse:{}, mae:{}\n\n'.format(mse, mae))

        # np.save(os.path.join(self.exp_dir, 'test_metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        # np.save(os.path.join(self.exp_dir, 'test_pred.npy'), preds)
        # np.save(os.path.join(self.exp_dir, 'test_true.npy'), trues)
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'MSPE': mspe
        }

    def apply_standing_statistics(self):
        self.target_model.train()
        self.target_model.apply(reset_batchnorm_statistics)
        
        with torch.no_grad():
            loader_iter = iter(self.train_loader)
            for _ in range(self.args.standing_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)
                
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self.get_decoder_input(batch_y)
                _ = self.get_output(self.target_model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
