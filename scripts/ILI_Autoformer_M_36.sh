python -u run.py \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --exp_name ILI_Autoformer_M_36_24 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7

python -u run.py \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --exp_name ILI_Autoformer_M_36_36 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7

python -u run.py \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --exp_name ILI_Autoformer_M_36_48 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7

python -u run.py \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --exp_name ILI_Autoformer_M_36_60 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7
