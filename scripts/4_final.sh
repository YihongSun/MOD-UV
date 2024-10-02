cuda_id=0
ExpName="example"
RndName="R3:Final"
NumEp=20
epoch_size=8000
seed=0
anno_dir="save/example/L2_pseudo_labels/"
data_dir="./data/waymo"

python3 moduv/train.py -n $ExpName --round_name $RndName --epochs $NumEp --epoch-size $epoch_size --cuda_id $cuda_id --seed $seed --data_dir $data_dir --anno_dir $anno_dir