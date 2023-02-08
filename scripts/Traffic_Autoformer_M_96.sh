python -u run.py \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --exp_name Traffic_Autoformer_M_96_96 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862

python -u run.py \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --exp_name Traffic_Autoformer_M_96_192 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862

python -u run.py \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --exp_name Traffic_Autoformer_M_96_336 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862

python -u run.py \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --exp_name Traffic_Autoformer_M_96_720 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862
