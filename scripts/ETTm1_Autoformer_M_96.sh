python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --exp_name ETTm1_Autoformer_M_96_96 \
  --model Autoformer \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7

python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --exp_name ETTm1_Autoformer_M_96_192 \
  --model Autoformer \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7

python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --exp_name ETTm1_Autoformer_M_96_336 \
  --model Autoformer \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7

python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --exp_name ETTm1_Autoformer_M_96_720 \
  --model Autoformer \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7
