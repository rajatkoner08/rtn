# Relation Transformer Network


This repository contains data and code for the papers [Relation Transformer Network](https://arxiv.org/abs/2004.06193) and [Scenes and Surroundings: Scene Graph Generation using Relation Transformer(ICML workshop,2020)
](https://arxiv.org/abs/2107.05448). This branch is specifically for GQA and the generated graph can be for visual question answering(VQA), please see our work [Graphhopper: Multi-Hop Scene Graph Reasoning for Visual Question Answering
(ISWC,2021)](https://arxiv.org/abs/2107.06325) for more details. If you like the paper, please cite our work:

### Bibtex

```
@article{koner2020relation,
  title={Relation transformer network},
  author={Koner, Rajat and Sinhamahapatra, Poulami and Tresp, Volker},
  journal={arXiv preprint arXiv:2004.06193},
  year={2020}
}
@article{koner2021scenes,
  title={Scenes and Surroundings: Scene Graph Generation using Relation Transformer},
  author={Koner, Rajat and Sinhamahapatra, Poulami and Tresp, Volker},
  journal={arXiv e-prints},
  pages={arXiv--2107},
  year={2021}
}
```
## Setup


0. At first, please install all the dependencies using ```pip install -r requirement.txt```. I recommend the [Anaconda distribution](https://repo.continuum.io/archive/). 
This version is trained on [Pytorch 1.4](https://pytorch.org/get-started/previous-versions/) for GQA, for other datasets (e.g., Visual Genome) please see the other branches.
1. Update the config file with the dataset paths and download all the files under ```data``` folder. Specifically:
    - Images of GQA under in ```data/gqa``` and its associated [scene graph](https://nlp.stanford.edu/data/gqa/sceneGraphs.zip) in ```data/gqa/graph```. 
2. Compile everything. run ```make``` in the main directory.
3. Installation of [MMdetection](https://github.com/open-mmlab/mmdetection) 
    ```shell
    mkdir mmdetection
    cd mmdetection
    git init
    git remote add -f origin <url_of_rtn> 
    git pull origin feature/sg # branch contains specific changes to adapt MMdet for GQA along with attribute prediction
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```
4. To generate valid classes, attribute,relation based on your need run ```dataloaders/load_gqa_from_json.py```

## GQA Object Detection
We have train [Detecto RS](https://github.com/open-mmlab/mmdetection/blob/master/configs/detectors/README.md) for GQA with ResNet-50 backbone, for better performance please train with SWIN-Transformer from MMDetection.
Trained object detector and configaration are avaliable at : [link](https://syncandshare.lrz.de/getlink/fiGT7TZboX9qtwcdBStj2oBr/)
## Training 
For training :```CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python models/train_rels.py -m predcls -nl_obj 3 -nl_edge 2 -attn_dim 2048 -ngpu 1 -ckpt PATH_to_mmdet_model -mmdet_config PATH_to_mmdet_config -save_dir PATH_to_save -nepoch 30 -obj_index_enc -highlight_sub_obj -n_head 12 -p 1000 -b 4 -dropout 0.25 -run_desc 3_2_gqa_full -l2 1e-5 -lr 5e-2 -normalized_roi -use_union_boxes -spo spo -use_extra_pos -train_obj_roi -dataset gqa -mmdet -pooling_dim 4096 -eval_method gt```
## Evaluation
For evaluation on VG (also you can download pre-trained weight of [RTN](https://syncandshare.lrz.de/getlink/fi8Q3pMt6yPo4J1w5fbpSUEn/) ): 
```CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python models/eval_rels.py -m predcls -nl_obj 3 -nl_edge 2 -attn_dim 2048 -ngpu 1 -ckpt PATH_to_checkpoints -obj_index_enc -highlight_sub_obj -n_head 12 -b 10 -normalized_roi -use_union_boxes -spo spo -use_extra_pos -train_obj_roi -pooling_dim 4096 -use_gap -use_word_emb -use_bias -test -save_dir PATH_to_save```
## Generation of Scene Graph for VQA
```CUDA_VISIBLE_DEVICES=0 python models/eval_inference.py -m predcls -nl_obj 3 -nl_edge 2 -attn_dim 2048 -ckpt PATH_to_checkpoints -obj_index_enc -highlight_sub_obj -n_head 12 -normalized_roi -use_union_boxes -spo spo -use_extra_pos -train_obj_roi -pooling_dim 4096 -use_gap -use_word_emb -use_bias -test -save_dir PATH_to_save```
It will save ground truth and predicted graph in a pickle file.
## help

Feel free to open an issue if you encounter trouble getting it to work!

# acknowledgment 
Part of the code is inspired from [rowanz/neural-motifs](https://github.com/rowanz/neural-motifs)

