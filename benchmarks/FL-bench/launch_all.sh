# cd ./data
# python generate_data.py -d cifar10 -c 3 -cn 20
# cd ../

# cd ./src/server
# # python apfl.py --model lenet5 --dataset cifar10 --save_model 1 --finetune_epoch 3 --global_epoch 500
# # python fedrep.py --model lenet5 --dataset cifar10 --save_model 1 --finetune_epoch 3 --global_epoch 500
# # python fedper.py --model lenet5 --dataset cifar10 --save_model 1 --finetune_epoch 3 --global_epoch 500
# # python ditto.py --model lenet5 --dataset cifar10 --save_model 1 --finetune_epoch 3 --global_epoch 500
# # python fedbn.py --model lenet5 --dataset cifar10 --save_model 1 --finetune_epoch 3 --global_epoch 500
# # python fedbabu.py --model lenet5 --dataset cifar10 --save_model 1 --finetune_epoch 3 --global_epoch 500
# # python perfedavg.py --model lenet5 --dataset cifar10 --save_model 1 --finetune_epoch 3 --global_epoch 500
# # python fedavg.py --model lenet5 --dataset cifar10 --save_model 1 --finetune_epoch 0 --global_epoch 500
# # python local.py --model lenet5 --dataset cifar10 --save_model 1 --finetune_epoch 0 --global_epoch 500
# python fedpn.py --model natpn --dataset cifar10 --save_model 1 --finetune_epoch 3 --global_epoch 500
# cd ../../


cd ./data
python generate_data.py -d svhn -c 3 -cn 20
cd ../

cd ./src/server
# python apfl.py --model lenet5 --dataset svhn --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedrep.py --model lenet5 --dataset svhn --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedper.py --model lenet5 --dataset svhn --save_model 1 --finetune_epoch 3 --global_epoch 500
# python ditto.py --model lenet5 --dataset svhn --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbn.py --model lenet5 --dataset svhn --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbabu.py --model lenet5 --dataset svhn --save_model 1 --finetune_epoch 3 --global_epoch 500
# python perfedavg.py --model lenet5 --dataset svhn --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedavg.py --model lenet5 --dataset svhn --save_model 1 --finetune_epoch 0 --global_epoch 500
# python local.py --model lenet5 --dataset svhn --save_model 1 --finetune_epoch 0 --global_epoch 500
python fedpn.py --model natpn --dataset svhn --save_model 1 --finetune_epoch 3 --global_epoch 500
cd ../../


cd ./data
python generate_data.py -d mnist -c 3 -cn 20
cd ../

cd ./src/server
# python apfl.py --model lenet5 --dataset mnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedrep.py --model lenet5 --dataset mnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedper.py --model lenet5 --dataset mnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python ditto.py --model lenet5 --dataset mnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbn.py --model lenet5 --dataset mnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbabu.py --model lenet5 --dataset mnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python perfedavg.py --model lenet5 --dataset mnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedavg.py --model lenet5 --dataset mnist --save_model 1 --finetune_epoch 0 --global_epoch 500
# python local.py --model lenet5 --dataset mnist --save_model 1 --finetune_epoch 0 --global_epoch 500
python fedpn.py --model natpn --dataset mnist --save_model 1 --finetune_epoch 3 --global_epoch 500
cd ../../


cd ./data
python generate_data.py -d fmnist -c 3 -cn 20
cd ../

cd ./src/server
# python apfl.py --model lenet5 --dataset fmnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedrep.py --model lenet5 --dataset fmnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedper.py --model lenet5 --dataset fmnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python ditto.py --model lenet5 --dataset fmnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbn.py --model lenet5 --dataset fmnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbabu.py --model lenet5 --dataset fmnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python perfedavg.py --model lenet5 --dataset fmnist --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedavg.py --model lenet5 --dataset fmnist --save_model 1 --finetune_epoch 0 --global_epoch 500
# python local.py --model lenet5 --dataset fmnist --save_model 1 --finetune_epoch 0 --global_epoch 500
python fedpn.py --model natpn --dataset fmnist --save_model 1 --finetune_epoch 3 --global_epoch 500
cd ../../


cd ./data
python generate_data.py -d medmnistA -c 3 -cn 20
cd ../

cd ./src/server
# python apfl.py --model lenet5 --dataset medmnistA --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedrep.py --model lenet5 --dataset medmnistA --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedper.py --model lenet5 --dataset medmnistA --save_model 1 --finetune_epoch 3 --global_epoch 500
# python ditto.py --model lenet5 --dataset medmnistA --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbn.py --model lenet5 --dataset medmnistA --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbabu.py --model lenet5 --dataset medmnistA --save_model 1 --finetune_epoch 3 --global_epoch 500
# python perfedavg.py --model lenet5 --dataset medmnistA --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedavg.py --model lenet5 --dataset medmnistA --save_model 1 --finetune_epoch 0 --global_epoch 500
# python local.py --model lenet5 --dataset medmnistA --save_model 1 --finetune_epoch 0 --global_epoch 500
python fedpn.py --model natpn --dataset medmnistA --save_model 1 --finetune_epoch 3 --global_epoch 500
cd ../../


cd ./data
python generate_data.py -d medmnistS -c 3 -cn 20
cd ../

cd ./src/server
# python apfl.py --model lenet5 --dataset medmnistS --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedrep.py --model lenet5 --dataset medmnistS --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedper.py --model lenet5 --dataset medmnistS --save_model 1 --finetune_epoch 3 --global_epoch 500
# python ditto.py --model lenet5 --dataset medmnistS --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbn.py --model lenet5 --dataset medmnistS --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbabu.py --model lenet5 --dataset medmnistS --save_model 1 --finetune_epoch 3 --global_epoch 500
# python perfedavg.py --model lenet5 --dataset medmnistS --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedavg.py --model lenet5 --dataset medmnistS --save_model 1 --finetune_epoch 0 --global_epoch 500
# python local.py --model lenet5 --dataset medmnistS --save_model 1 --finetune_epoch 0 --global_epoch 500
python fedpn.py --model natpn --dataset medmnistS --save_model 1 --finetune_epoch 3 --global_epoch 500
cd ../../


cd ./data
python generate_data.py -d medmnistC -c 3 -cn 20
cd ../

cd ./src/server
# python apfl.py --model lenet5 --dataset medmnistC --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedrep.py --model lenet5 --dataset medmnistC --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedper.py --model lenet5 --dataset medmnistC --save_model 1 --finetune_epoch 3 --global_epoch 500
# python ditto.py --model lenet5 --dataset medmnistC --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbn.py --model lenet5 --dataset medmnistC --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedbabu.py --model lenet5 --dataset medmnistC --save_model 1 --finetune_epoch 3 --global_epoch 500
# python perfedavg.py --model lenet5 --dataset medmnistC --save_model 1 --finetune_epoch 3 --global_epoch 500
# python fedavg.py --model lenet5 --dataset medmnistC --save_model 1 --finetune_epoch 0 --global_epoch 500
# python local.py --model lenet5 --dataset medmnistC --save_model 1 --finetune_epoch 0 --global_epoch 500
python fedpn.py --model natpn --dataset medmnistC --save_model 1 --finetune_epoch 3 --global_epoch 500
cd ../../
