python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --exp_name ETTh1_Autoformer+EMA_M_96_96 \
  --model Autoformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --use_ema \
  --lr 0.0003 \
  --lradj no \
  --start_iter 100

python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --exp_name ETTh1_Autoformer+EMA_M_96_192 \
  --model Autoformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --use_ema \
  --lr 0.0003 \
  --lradj no \
  --start_iter 100

python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --exp_name ETTh1_Autoformer+EMA_M_96_336 \
  --model Autoformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --use_ema \
  --lr 0.0003 \
  --lradj no \
  --start_iter 100

python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --exp_name ETTh1_Autoformer+EMA_M_96_720 \
  --model Autoformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --use_ema \
  --lr 0.0003 \
  --lradj no \
  --start_iter 100
