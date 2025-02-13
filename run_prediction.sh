#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80000M
#SBATCH --time=1:00:00
#SBATCH --account=def-hup-ab

module load StdEnv/2020 gcc/9.3.0 cuda/11.7 arrow/9.0.0 python/3.9
source /home/gzhou54/projects/def-hup-ab/gzhou54/my_model_env/bin/activate

nvidia-smi

python run_model.py \
        --smiles_graph_weight 0.4 \
        --smiles_fp_weight 0.4 \
        --graph_fp_weight 0.2 \
        --graph_model 'bi_graphormer_no_mpnn' \
        --model_path 'checkpoint.ckpt' \
        --dataset_name zinc \
        --prediction_test True