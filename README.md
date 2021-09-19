# Relation Transformer Network


This repository contains data and code for the papers [Relation Transformer Network](https://arxiv.org/abs/2004.06193) and [Scenes and Surroundings: Scene Graph Generation using Relation Transformer(ICML workshop,2020)
](https://arxiv.org/abs/2107.05448). This repository can also be used as a scene graph generator for visual question answering(VQA), please see our work [Graphhopper: Multi-Hop Scene Graph Reasoning for Visual Question Answering
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
# Setup


0. At first, please install all the dependencies using ```pip install -r requirement.txt```. I recommend the [Anaconda distribution](https://repo.continuum.io/archive/). 
This version is trained on [Pytorch 0.4](https://pytorch.org/get-started/previous-versions/) only with Visual Genome, for other datasets (e.g., GQA) please see the other branches (accuracy may differ slightly).
1. Update the config file with the dataset paths and download all the files under ```data``` folder. Specifically:
    - Visual Genome (the VG_100K folder, image_data.json, VG-SGG.h5, and VG-SGG-dicts.json). See data/stanford_filtered/README.md for the steps I used to download these.
2. Compile everything. run ```make``` in the main directory: this compiles the Bilinear Interpolation operation for the RoI align as well as the non-maximum suppression(NMS).

# Training 
For training :```CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python models/train_rels.py -m predcls -nl_obj 3 -nl_edge 2 -attn_dim 2048 -ngpu 1 -ckpt Your_Checkpoint_Path -save_dir Your_Save_Dir -nepoch 30 -obj_index_enc -highlight_sub_obj -n_head 12 -b 20 -dropout 0.25 -l2 1e-5 -lr 5e-2 -normalized_roi -use_union_boxes -spo spo -use_extra_pos -train_obj_roi```
#Evaluation
For evaluation on VG (also you can download pre-trained weight of [RTN](https://syncandshare.lrz.de/getlink/fi8Q3pMt6yPo4J1w5fbpSUEn/) ): 
```CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python models/eval_rels.py -m predcls -nl_obj 3 -nl_edge 2 -attn_dim 2048 -ngpu 1 -ckpt ckpt_path -obj_index_enc -highlight_sub_obj -n_head 12 -b 10 -normalized_roi -use_union_boxes -spo spo -use_extra_pos -train_obj_roi -pooling_dim 4096 -use_gap -use_word_emb -use_bias -test -save_dir data/checkpoints```

# help

Feel free to open an issue if you encounter trouble getting it to work!

# acknowledgment 
Part of the code is inspired from [rowanz/neural-motifs](https://github.com/rowanz/neural-motifs)

