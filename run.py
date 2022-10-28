import argparse
import os
import torch
from exp.exp_main import ExpMain
import numpy as np
from utils.logger import make_logger
from utils.recorder import Recorder
from utils.tools import fix_seed
import json

fix_seed()

parser = argparse.ArgumentParser(description='Time Series Forecasting')

# basic config
parser.add_argument('--exp_name', type=str, required=True, default='test', help='exp_name')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')
parser.add_argument('--result_dir', type=str, default='./save/results', help='result path')
parser.add_argument('--checkpoint_dir', type=str, default='./save/checkpoints', help='checkpoint path')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--itr', type=int, default=3, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adamw'], help='optimizer')
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='optimizer learning rate decay factor')
parser.add_argument('--loss', type=str, default='MSE')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# EMA
parser.add_argument('--use_ema', action='store_true')
parser.add_argument('--epsilon', type=float, default=0.01)
parser.add_argument('--moving_average_decay', type=float, default=0.99)
parser.add_argument('--standing_steps', type=int, default=100)
parser.add_argument('--start_iter', type=int, default=300)
parser.add_argument('--ema_loss', type=str, default='BDFMSE')
parser.add_argument('--ema_eval_model', type=str, default='target', choices=['source', 'target'])

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


os.makedirs(os.path.join(args.checkpoint_dir, args.exp_name), exist_ok=True)

with open(os.path.join(args.checkpoint_dir, args.exp_name, 'arguments.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

logger = make_logger(os.path.join(args.checkpoint_dir, args.exp_name, f'{args.exp_name}.log'))

logger.info('Args in experiment:')
logger.info(args)

valid_recorder = Recorder(args, 'valid_metrics')
test_recorder = Recorder(args, 'test_metrics')

Exp = ExpMain

for ii in range(args.itr):
    exp = Exp(args, ii, logger)  # set experiments
    logger.info('>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train()

    logger.info('>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    valid_metric_dict = exp.test('valid')
    valid_recorder.writerow(ii, valid_metric_dict)
    test_metric_dict = exp.test('test')
    test_recorder.writerow(ii, test_metric_dict)

    torch.cuda.empty_cache()

valid_recorder.write_statistics()
valid_recorder.close()

test_recorder.write_statistics()
test_recorder.close()
