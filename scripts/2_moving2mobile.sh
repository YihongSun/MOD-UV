cuda_id=0
ExpName="example"
RndName="R1:M2M"
NumEp=3
epoch_size=8000
seed=0
anno_dir="save/example/L0_pseudo_labels/"
data_dir="./data/waymo"

python3 moduv/train.py -n $ExpName --round_name $RndName --epochs $NumEp --epoch-size $epoch_size --cuda_id $cuda_id --seed $seed --data_dir $data_dir --anno_dir $anno_dir
python3 moduv/pseudo_labels/save_pred.py --data_dir $data_dir --m2m_ckpt ./save/$ExpName/$RndName/weights_$(printf "%02d" "$((NumEp-1))").pth  --cuda_id $cuda_id