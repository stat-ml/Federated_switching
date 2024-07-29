import pandas as pd

method = "FedBN"
table_save_name = f'{method}_benchmark_compact_results.csv'

print(method)
comp_results = pd.read_csv(table_save_name)
grouped_df = comp_results.groupby('dataset')[[col for col in comp_results.columns if col not in  ['client_id', 'dataset']]].mean()
print(grouped_df)

# import torch
# print(torch.load("FL-bench/out/FedRep/cifar10_100_lenet5.pt")['classifier.bias'])