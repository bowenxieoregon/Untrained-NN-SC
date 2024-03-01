for i in {1..1}
do
  python run_dip_hun.py --rho 0.05 --channel 256 --layers 6 --filters 3 --alpha 0.4 --patience 2 --lr 0.01 --inner 80 --model saved_model/model_7.pth  --order 4 --data_type strong_6_100_clear&
done

for i in {1..1}
do
  python run_dip_hun.py --rho 0.1 --channel 256 --layers 6 --filters 3 --alpha 0.4 --patience 2 --lr 0.01 --inner 80 --model saved_model/model_8.pth  --order 5 --data_type strong_6_100_clear&
done

for i in {1..1}
do
  python run_dip_hun.py --rho 0.15 --channel 256 --layers 6 --filters 3 --alpha 0.4 --patience 2 --lr 0.01 --inner 80 --model saved_model/model_9.pth  --order 6 --data_type strong_6_100_clear&
done

for i in {1..1}
do
  python run_dip_hun.py --rho 0.05 --channel 256 --layers 6 --filters 3 --alpha 0.25 --patience 2 --lr 0.01 --inner 80 --model saved_model/model_10.pth  --order 7 --data_type strong_6_100_clear&
done

for i in {1..1}
do
  python run_dip_hun.py --rho 0.1 --channel 256 --layers 6 --filters 3 --alpha 0.25 --patience 2 --lr 0.01 --inner 80 --model saved_model/model_11.pth  --order 8 --data_type strong_6_100_clear&
done

for i in {1..1}
do
  python run_dip_hun.py --rho 0.15 --channel 256 --layers 6 --filters 3 --alpha 0.25 --patience 2 --lr 0.01 --inner 80 --model saved_model/model_12.pth  --order 9 --data_type strong_6_100_clear&
done