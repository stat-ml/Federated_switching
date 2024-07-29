cd ./src/server

# CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset mnist \
#  --nat_pn_backbone lenet5 \
#  --batch_size 64 \
#  --embedding_dim 16 \
#  --density_model_type flow \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.001 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 20 \
#  --global_epoch 200 \
#  --local_epoch 1 \
#  --local_batchs 5 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.001 \
#  --momentum 0.0 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
#  --use_wandb true \
#  --optimizer_name Adam \

# CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset fmnist \
#  --nat_pn_backbone lenet5 \
#  --batch_size 64 \
#  --embedding_dim 16 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.01 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 20 \
#  --global_epoch 200 \
#  --local_epoch 1 \
#  --local_batchs 5 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.001 \
#  --momentum 0.0 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
#  --use_wandb true \
#  --optimizer_name Adam \


# CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset medmnistS \
#  --nat_pn_backbone lenet5 \
#  --batch_size 64 \
#  --embedding_dim 16 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.01 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 20 \
#  --global_epoch 200 \
#  --local_epoch 1 \
#  --local_batchs 5 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.001 \
#  --momentum 0.0 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
#  --use_wandb true \
#  --optimizer_name Adam \


 CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset medmnistA \
 --seed 41 \
 --density_model_type flow \
 --join_ratio 0.1 \
 --nat_pn_backbone lenet5 \
 --batch_size 32 \
 --embedding_dim 16 \
 --loss_name bayessian \
 --loss_log_prob_weight 0.01 \
 --loss_entropy_weight 0.0 \
 --loss_embeddings_weight 0.0 \
 --finetune_in_the_end 20 \
 --global_epoch 500 \
 --local_epoch 1 \
 --local_batchs 5 \
 --stop_grad_logp true \
 --stop_grad_embeddings false \
 --server_cuda 1 \
 --client_cuda 1 \
 --local_lr 0.001 \
 --momentum 0.0 \
 --eval_test 0 \
 --save_metrics 0 \
 --save_fig 0 \
 --save_log 0 \
 --use_wandb true \
 --optimizer_name Adam \


#  CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset medmnistC \
#  --nat_pn_backbone lenet5 \
#  --batch_size 64 \
#  --embedding_dim 16 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.01 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 20 \
#  --global_epoch 200 \
#  --local_epoch 1 \
#  --local_batchs 5 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.001 \
#  --momentum 0.0 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
#  --use_wandb true \
#  --optimizer_name Adam \

