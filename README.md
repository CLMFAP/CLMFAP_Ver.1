Before pretraining, please prepare the dataset to be used for pretraining and create a folder named data. Place the pretraining dataset inside the data folder.

Please preprocess the pretraining dataset first. Use rdkit to convert each SMILES molecule into its canonical form, then retain only the SMILES while keeping the first column as the index.

Before fine-tuning, please prepare the dataset to be used for fine-tuning and split it into train, valid, and test files. Create a folder named after the dataset and place the train, valid, and test files inside it. Then, modify the configuration file to add the dataset path, the measure name, and the number of classification categories.

For fine-tuning, you need to provide the pretrained model path parameter "model_path" and the dataset parameter "dataset_name".

We provide pip package list as a reference for the Python environment. Below are some of the main packages along with installation tips for certain packages.

How to use this model to predict any molecules:
1, Create Environment:
1.1 For local desktop environment, please refer the 'local_pip_list.txt'
1.2 For the compute canada, please refer the 'canada_pip_list.txt'
1.3 main packages:
1.3.1
pip install transformers
pip install rdkit
pip install torch-scatter
pip install orch-sparse
pip install torch-cluster
pip install torch-spline-conv
pip install torch-geometric 
pip install pytorch_lightning
pip install pytorch-fast-transformers

1.3.2 install apex:
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 22.04-dev
export CUDA_HOME=/usr/local/cuda-11.7 # Change to your cuda path
export TORCH_CUDA_ARCH_LIST="7.0" # Depend on cuda version

change the downloaded 'apex/amp/_initialize.py' line 2 like this:
#from torch._six import string_classes
string_classes = str

python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

1.3.3 install pytorch-fast-transformers:
https://github.com/idiap/fast-transformers/issues/117

1.3.4 install visualizer
git clone https://github.com/luo3300612/Visualizer.git
cd Visualizer
pip install bytecode
python setup.py install

2, Prepare dataset:
2.1 Process the data to only keep smiles and name the file test.csv with head "smiles".
2.2 Move the file to data/zinc folder.

3, Predicting:
3.1 Run the `run_prediction.sh` for local desktop environment.
3.2 Run the `run_prediction_remote.sh` for the compute canada.
3.3 The prediction result will be saved at "result/test_result/results_test.csv"
