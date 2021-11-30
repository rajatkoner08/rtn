#!/usr/bin/env bash
#
#SBATCH --job-name torch-test
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:1

# debug info
hostname
which python3
nvidia-smi

export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/usr/local/cuda-10.1/
env

# venv
#python3 -m venv ./venv/test
#source ./venv/test/bin/activate
#pip install -U pip setuptools wheel
#pip install -U torch torchvision

source ~/.bashrc
source activate torch1.4

# test cuda
python -c "import torch; print(torch.cuda.device_count())"

#python setup.py install

#python -m pip install mmcv-full==latest+torch1.4.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

#python -c 'import mmdet'

python tools/train.py configs/faster_rcnn/faster_rcnn_r101_fpn_2x_gqa.py

# download example script for CNN training
#SRC=src/${SLURM_ARRAY_JOB_ID}
#mkdir -p ${SRC}
#wget https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py -O ${SRC}/torch-test.py
#cd ${SRC}

# train
#python3 ./torch-test.py
