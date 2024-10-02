cuda_id=0
ExpName="example"
RndName="R2:L2S"
NumEp=20
epoch_size=8000
seed=0
anno_dir="save/example/L1_pseudo_labels/"
data_dir="./data/waymo"

# Note that the two trainings can be executed in parallel as independent jobs. We implement sequential training for simplicity.
python3 moduv/train.py -n $ExpName --round_name $RndName:L --epochs $NumEp --epoch-size $epoch_size --cuda_id $cuda_id --seed $seed --data_dir $data_dir --anno_dir $anno_dir --sjitter_rate 0.0 
python3 moduv/train.py -n $ExpName --round_name $RndName:S --epochs $NumEp --epoch-size $epoch_size --cuda_id $cuda_id --seed $seed --data_dir $data_dir --anno_dir $anno_dir --sjitter_rate 1.0 --sjitter_min 0.25 --sjitter_max 0.25

python3 moduv/pseudo_labels/save_agg_pred.py --data_dir $data_dir --cuda_id $cuda_id --l2s_l_ckpt ./save/$ExpName/$RndName:L/weights_$(printf "%02d" "$((NumEp-1))").pth --l2s_s_ckpt ./save/$ExpName/$RndName:S/weights_$(printf "%02d" "$((NumEp-1))").pth