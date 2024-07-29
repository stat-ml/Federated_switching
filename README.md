---

# Complementary Code for IJCAI 2024 Submission

This repository provides supplemental material for our IJCAI 2024 paper submission titled **"Dirichlet-based Uncertainty Quantification for Personalized Federated Learning with Improved Posterior Networks"**. The codebase has been substantially derived from the publicly available [Federated Learning benchmark](https://github.com/KarhouTam/FL-bench) and the [Posterior Networks/Natural Posterior Networks code](https://github.com/borchero/natural-posterior-network).

## Code Structure

There are two folders in the root. Both are heavily based on the repository mentioned above.

### FedPN

In the ``FedPN'' folder, there is a code for training FedPN model and for conduction corresponding experiments.

The root directory contains several bash scripts that initiate experiments and save the resulting models. After setting up a virtual environment with the necessary packages specified in `requirements.txt` and generating the requisite data, these scripts can be executed to train the models.

A distinct notebook is provided for each experiment under the 'experiments' section. These notebooks are instrumental in generating all the figures and tables presented in the paper.

## Data Generation

To generate the necessary data, follow the steps outlined below:

1. Navigate to the directory `FL-bench/data/utils`. Depending on the experiment you wish to conduct, execute the appropriate command:
   - For heterogeneous training (referenced in Tables), run the following command for each dataset:
     ```
     python run.py -d cifar10 -c 3 -cn 20
     ```
     Substitute `cifar10` with the name of your chosen dataset.
     
   - For centralized training with a noisy (aleatoric) dataset, execute:
     ```
     python run.py -d noisy_mnist --iid 1 -cn 1
     ```

2. The data for the toy experiment will be automatically generated when running the `toy_script.sh` script.

Once the data is generated, you may commence model training using the `runall_script.sh` script. This will save the models into `out/FedAvg/`.

You can then use this path to run the notebooks and reproduce the results.

### Benchmarks


This folder is pretty much the same as another one, as both are based on the same codebase. 

Within ``FL-bench'' repository there is a training script ``launch_all.sh'' for each of the models (you can choose whichever dataset you want).

Then, in the root, there are two files: ``load_models.py'' and ``postproc.py''. They will produce data for the tables.

---