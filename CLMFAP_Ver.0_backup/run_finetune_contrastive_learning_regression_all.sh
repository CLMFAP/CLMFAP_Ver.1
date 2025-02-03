# !/bin/bash
# SBATCH --nodes=1
# SBATCH --gpus-per-node=a100:1
# SBATCH --ntasks-per-node=1
# SBATCH --mem=80000M
# SBATCH --time=48:00:00

nvidia-smi
# module load StdEnv/2020 gcc/9.3.0 cuda/11.7 arrow/9.0.0 python/3.9

python finetune_contrastive_learning_regression.py \
        --smiles_graph_weight 0.4 \
        --smiles_fp_weight 0.4 \
        --graph_fp_weight 0.2 \
        --dataset_name ALL