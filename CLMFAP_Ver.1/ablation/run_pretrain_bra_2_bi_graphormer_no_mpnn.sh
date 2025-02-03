#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8000M
#SBATCH --time=00:20:00
#SBATCH --account=def-hup-ab

module load StdEnv/2020 gcc/9.3.0 cuda/11.7 arrow/9.0.0 python/3.9
source /home/gzhou54/projects/def-hup-ab/gzhou54/my_env_2/bin/activate

nvidia-smi

pip list
which nvcc
export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.7.0
export CUDA_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.7.0
export TORCH_USE_CUDA_DSA=1
export TORCH_CUDA_ARCH_LIST="8.0"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.current_device())"
python -c "import torch; print(torch.cuda.device(0))"
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import torch; print(torch.version.cuda)"
#python -c "import torch; print(torch.get_device_capability(0))"
python pretrain_contrastive_learning_test_data_loss.py \
        --smiles_graph_weight 0.4 \
        --smiles_fp_weight 0.4 \
        --graph_fp_weight 0.2 \
        --graph_model 'bi_graphormer_no_mpnn' \
        --bra_size 2