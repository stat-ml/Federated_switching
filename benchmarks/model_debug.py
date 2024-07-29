import sys
sys.path.insert(0, './FL-bench')
sys.path.insert(0, './FL-bench/src/server')

from src.server.local import LocalServer
from src.server.fedavg import get_fedavg_argparser
import torch


dataset_name = 'cifar10'


# print(torch.load(f"./FL-bench/out/Local-only/{dataset_name}_100_lenet5.pt")[0][0][0][0])
print(type(torch.load(f"./FL-bench/out/APFL/{dataset_name}_100_lenet5.pt")))

# args = get_fedavg_argparser().parse_args()
# args.external_model_params_file = f"FL-bench/out/Local/{dataset_name}_100_lenet5.pt"
# args.finetune_epoch = 3
# args.global_epoch = 0
# args.test_gap = 1
# args.dataset = dataset_name

# model = LocalServer(args=args)
# model = APFLServer(args=args)
# model.run()

# model.test()
# out = model.test_results

# print(out)