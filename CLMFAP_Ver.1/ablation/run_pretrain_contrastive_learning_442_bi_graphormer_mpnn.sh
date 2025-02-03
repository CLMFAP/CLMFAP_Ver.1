# !/bin/bash
# SBATCH --nodes=1
# SBATCH --gpus-per-node=v100l:1
# SBATCH --ntasks-per-node=1
# SBATCH --mem=80000M
# SBATCH --time=48:00:00
# SBATCH --account=def-hup-ab

nvidia-smi
echo 'Emm, Grey Cat!'

# module load StdEnv/2020 gcc/9.3.0 cuda/11.7 arrow/9.0.0 python/3.9
# source /home/gzhou54/projects/def-hup-ab/gzhou54/molformer_env/bin/activate

#cd ../../apex
#export TORCH_CUDA_ARCH_LIST="7.0"
#pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

python pretrain_contrastive_learning_test_data_loss.py \
        --smiles_graph_weight 0.8 \
        --smiles_fp_weight 0.1 \
        --graph_fp_weight 0.1 \
        --graph_model 'bi_graphormer_mpnn'