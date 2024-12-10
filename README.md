# Preference Optimization for Molecule Synthesis with Conditional Residual Energy-based Models

This repository contains an implementation of ["Preference Optimization for Molecule Synthesis with Conditional Residual Energy-based Models"](https://openreview.net/pdf?id=oLfq1KKneW), which is a general and principled framework for controllable molecule synthetic route generation.  

## Warning 
***We only use one route for each molecule in the training dataset for model training!!!***

## Dropbox 

We provide the starting material file in dropbox, you can download this file via: 
https://www.dropbox.com/scl/fi/j3kh641irxtpbrnjnmoop/zinc_stock_17_04_20.hdf5?rlkey=zqbymj13skpdqlswu2uvji1sq&st=c1805gz0&dl=0
Please move this file into the root folder.  

## (I). REBM Training
**Note: We have provided the energy function model in the Neuralsym, GLN, GraphRetro, Transformer, and FusionRetro folders for inference. However, we also provide the following commands for energy function training if you wish to reproduce the training process.**

### 1. Train the forward model

While we provide the trained forward model, you can also use the following commands to reproduce the training process.
```bash
cp train_dataset.json valid_dataset.json test_dataset.json Forward_model/data
cd Forward_model

#Data Process

#Canolize dataset
python to_canolize.py --dataset train  
python to_canolize.py --dataset valid  
python to_canolize.py --dataset test

#Json to txt
python get_dataset.py --phase train  
python get_dataset.py --phase valid  
python get_dataset.py --phase test

#Get augmented training data
python get_augmented_reaction.py --phase train
python train.py --batch_size 256
```

Or Skip the forward model training, move the already provided forward model file to the Neuralsym folder
```bash
mv models/forward_model.pkl ../Neuralsym/
cd ../
```

### 2. Sample synthetic routes $\mathcal{D}$ from the training dataset using Neuralsym+Retro*-0

(1) We provide the following commands for Neuralsym training if you wish to reproduce the training process.
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 Neuralsym/  
cd Neuralsym  

#Data Process
python prepare_data.py  

#Train
bash train.sh  
```

However, you can skip the Neuralsym training process since we have provided the trained Neuralsyn model in dropbox, you can download this file via:
https://www.dropbox.com/scl/fi/le78g7kbv1ufhznh2sair/Highway_42_depth0_dim300_lr1e3_stop2_fac30_pat1.pth.tar?rlkey=1j6sz9zoqm4hz1qwwgwssrj9e&st=8jood7d8&dl=0
Please move this file into Neuralsym/checkpoint folder. 

(2) Once you have obtained the Neuralsym model, you can use it with Retro*-0 (beam search) to sample synthetic routes $\mathcal{D}$ from the training dataset. Besides, you need to use forward model to rank routes.
```bash
python get_rerank_dataset.py
```

We also provid the dataset $\mathcal{D}$ (candidate_dataset.json), you can mv this file to the Energy_function folder directly and skip the samping process.
```bash
mv candidate_dataset.json ../Energy_function/
cd ..
```

### 3. Train the enegy function
Once you have obtained $\mathcal{D}$ (candidate_dataset.json), you can use it to train the energy function. But we first pretrain the energy function on the [target -> starting material] task. 
```bash
#copy dataset to Energy_folder
cp train_dataset.json Energy_function/
cd Energy_function

#dataset canolize
python to_canolize.py --dataset train  

#get pretrained dataset
python get_pretrain_dataset.py
python train.py --batch_size 32  
```
Or Skip the pre-training since we have provided the pretrained reward model

Then train the energy function
```bash
python reward_tune_schedule.py --epochs 40 --lr 5e-5
```

After completing steps 1, 2, and 3, you will have a well-trained energy function model. However, since we have provided our trained energy function model, you can skip these steps.
```bash
cp reward_models/reward_model.pkl ../Neuralsym
cp reward_models/reward_model.pkl ../GLN
cp reward_models/reward_model.pkl ../Transformer
cp reward_models/reward_model.pkl ../FusionRetro
cp reward_models/reward_model.pkl ../GraphRetro
cd ..
```

## (II). REBMRetro Inference

### Neuralsym

Train evalution function for Retro* search 
```bash
cd Neuralsym
#Retro Star Search
python get_reaction_cost.py  
python get_molecule_cost.py  
python value_mlp.py  
#We also provide value_mlp.pkl, you can skip the above commands
```
Or you can skip the above since we have provided the evaluation function model file (value_mlp.pkl)

REBM Inference
```bash
#Retro Star Zero Search
python rerank.py  

#Retro Star Search
python rarank_star.py  
```

### GLN

```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 GLN/gln/  
cd GLN  
pip install -e .  
cd gln  

#Data Process 
python process_data_stage_1.py -save_dir data  

python process_data_stage_2.py -save_dir data -num_cores 12 -num_parts 1 -fp_degree 2 -f_atoms data/atom_list.txt -retro_during_train False $@  

python process_data_stage_2.py -save_dir data -num_cores 12 -num_parts 1 -fp_degree 2 -f_atoms data/atom_list.txt -retro_during_train True $@  

#Train
bash run_mf.sh schneider  

# We select the model with the performance on all routes in the validation dataset
```

REBM Inference
```bash
#Retro Star Zero Search
python rerank.py -save_dir data -f_atoms data/atom_list.txt -gpu 0 -seed 42 -beam_size 5 -epoch_for_search 100  

#Retro Star Search
python get_reaction_cost.py -save_dir data -f_atoms data/atom_list.txt -gpu 0 -seed 42 -beam_size 10 -epoch_for_search 100  
python get_molecule_cost.py  
python value_mlp.py  
#We also provide value_mlp.pkl, you can skip the above commands
python rerank_star.py -save_dir data -f_atoms data/atom_list.txt -gpu 0 -seed 42 -beam_size 5 -epoch_for_search 100   
```


### GraphRetro
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 GraphRetro/datasets/uspto-50k  
cd GraphRetro  
export SEQ_GRAPH_RETRO=$(pwd)  
python setup.py develop  

#Data Process
mv datasets/uspto-50k/valid_dataset.json datasets/uspto-50k/eval_dataset.json  
python json2csv.py  
python data_process/canonicalize_prod.py --filename train.csv  
python data_process/canonicalize_prod.py --filename eval.csv  
python data_process/canonicalize_prod.py --filename test.csv  
python data_process/parse_info.py --mode train  
python data_process/parse_info.py --mode eval  
python data_process/parse_info.py --mode test  
python data_process/core_edits/bond_edits.py  
python data_process/lg_edits/lg_classifier.py  
python data_process/lg_edits/lg_tensors.py  

#Train
python scripts/benchmarks/run_model.py --config_file configs/single_edit/defaults.yaml  
python scripts/benchmarks/run_model.py --config_file configs/lg_ind/defaults.yaml  
# We select the model by the original' code's setting  

#We also provide model files, you can skip the above commands
```

REBM Inference
```bash
#Retro Star Zero Search
python scripts/eval/rerank.py --beam_size 5 --edits_exp SingleEdit_20220823_044246 --lg_exp LGIndEmbed_20220823_04432 --edits_step best_model --lg_step best_model --exp_dir models 

#Retro Star Search
python scripts/eval/get_reaction_cost.py --beam_size 10 --edits_exp SingleEdit_20220823_044246 --lg_exp LGIndEmbed_20220823_04432 --edits_step best_model --lg_step best_model --exp_dir models  
python scripts/eval/get_molecule_cost.py  
python scripts/eval/value_mlp.py  
#We also provide value_mlp.pkl, you can skip the above commands
python scripts/eval/rerank_star.py --beam_size 5 --edits_exp SingleEdit_20220823_044246 --lg_exp LGIndEmbed_20220823_04432 --edits_step best_model --lg_step best_model --exp_dir models   
```


### Transformer  
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 Transformer/  
cd Transformer  

#Data Process  
python to_canolize.py --dataset train  
python to_canolize.py --dataset valid  
python to_canolize.py --dataset test  

#Train  
python train.py --batch_size 32 --epochs 2000  
# We select the model with the performance on the first 100 routes in the validation dataset

#We also provide model.pkl, you can skip the above commands
```

REBM Inference
```bash
#Retro Star Zero Search
python rerank.py  --beam_size 5  

#Retro Star Search
python get_reaction_cost.py  
python get_molecule_cost.py  
python value_mlp.py  
#We also provide value_mlp.pkl, you can skip the above commands
python rerank_star.py --beam_size 5  
```


### FusionRetro 
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 FusionRetro/  
cd FusionRetro  

#Data Process  
python to_canolize.py --dataset train  
python to_canolize.py --dataset valid  
python to_canolize.py --dataset test  

#Initial Train
python train.py --batch_size 64 --epochs 3000  
# After 3000 epochs, We set global_step to 1000000 and continue to train the model (3000th epoch's model paramater) with 1000 epochs  
#Continue Train
python train.py --batch_size 64 --continue_train --epochs 1000
# We select the model with the performance on the first 100 routes in the validation dataset

#We also provide model.pkl, you can skip the above commands
```

REBM Inference
```bash
#Retro Star Zero Search
python rerank.py  --beam_size 5  

#Retro Star Search
python get_reaction_cost.py  
python get_molecule_cost.py  
python value_mlp.py  
#We also provide value_mlp.pkl, you can skip the above commands
python rerank_star.py --beam_size 5  
```

## Reference  
FusionRetro: https://github.com/SongtaoLiu0823/FusionRetro  


## Citation
```
@inproceedings{liu2024preference,
  title={Preference Optimization for Molecule Synthesis with Conditional Residual Energy-based Models},
  author={Liu, Songtao and Dai, Hanjun and Zhao, Yue and Liu, Peng},
  booktitle={International Conference on Machine Learning},
  year={2024},
}
```
