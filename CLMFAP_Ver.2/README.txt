Before pretraining, please prepare the dataset to be used for pretraining and create a folder named data. Place the pretraining dataset inside the data folder.

Please preprocess the pretraining dataset first. Use rdkit to convert each SMILES molecule into its canonical form, then retain only the SMILES while keeping the first column as the index.

Before fine-tuning, please prepare the dataset to be used for fine-tuning and split it into train, valid, and test files. Create a folder named after the dataset and place the train, valid, and test files inside it. Then, modify the configuration file to add the dataset path, the measure name, and the number of classification categories.

For fine-tuning, you need to provide the pretrained model path parameter "model_path" and the dataset parameter "dataset_name".

We provide pip package list as a reference for the Python environment. Below are some of the main packages along with installation tips for certain packages.

# Create conda environment:
conda create -n cl_mfap_model_env python=3.9
conda activate cl_mfap_model_env

# Below is an example installation with CUDA 11.7 and python 3.9. 
# For other versions, you need to adjust the package versions accordingly.
# For more torch VS cuda: https://pytorch.org/get-started/previous-versions/
pip install torch==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torchvision==0.15.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install setuptools==60.2.0

# Install apex
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 22.04-dev
export CUDA_HOME=/usr/local/cuda-11.7 # location of my cuda install
export TORCH_CUDA_ARCH_LIST="8.0"  # I'm compiling for use on A100s

#comment out the line 2 in 'apex/amp/_initialize.py' and add 'string_classes = str', like this:
#from torch._six import string_classes
string_classes = str

python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

pip install transformers==4.6.0
pip install rdkit==2022.03.3
pip install torch_geometric==2.3.1
pip install pytorch_lightning==1.1.5
pip install pytorch-fast-transformers==0.4.0
pip install pandas==1.2.4
pip install ogb

# Install Visualizer
git clone https://github.com/luo3300612/Visualizer.git
cd Visualizer
pip install bytecode
python setup.py install

pip install pip==24.0
pip install fairseq==0.12.2
pip install einops
pip install pandas==1.4.0
pip install numpy==1.25.2