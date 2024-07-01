# BERT-Pruning
## Paper
This repository contains the code for the report about Structure pruning on BERT : [BERT Pruning](https://ambroise012.github.io/files/BertPruning.pdf)


## Requirements

```bash
pip install -r requirements.txt
```
You also will need to add these 2 files (dev-v1.1.json and train-v1.1.json) :
[download here](https://drive.google.com/drive/folders/1KYW1smeuWviZuDCwW0Qf4JNejQf_SNsh?usp=sharing) 

Add the SQUAD folder containing these 2 files to the root of the project. 

### Tree Structure

```
.
├── BERT-Pruning            
│   ├── graph
│       └── ... graph_.py                             
│   ├── images
|       ├── adapter
│           └── ... graph.png                             
│       ├── adapter.py   
│   ├── job
|        ├── adapter_no_distill
│             └── ... .job                             
│        ├── eval
│             └── ... .job                             
│        ├── fine_tune
│             └── ... .job                                       
│        └── train
│             └── ... .job
│   ├── models
│       ├── __pycache__         
│       ├── __init__.py
│       ├── adapter.py        
│       ├── l0_module.py
│       ├── model_args.py
│       ├── modeling_bert.py        
│       ├── modeling_bert_adapter.py
│       └── modeling_roberta.py                                 
│   ├── trainer
|       ├── __pycache__
│       ├── train_squad.py      
│       └── trainer_adapter.py
│   ├── utils
│       ├── __pycache__         
│       ├── __init__.py
│       ├── utils.py
│       ├── cofi_utils.py        
│       ├── utils_squad.py
│       └── utils_squad_evaluate.py
|   ├── constantes.csv
|   ├── eval_metrics_adapter.csv
|   ├── evaluation.py
|   ├── glue.py
|   ├── loss.csv
|   ├── run_adapter.py
|   ├── run_glue_prune.py
|   ├── run_squad.py           
|   ├── requirements.txt                
|   ├── README.md                  
```



You can also download the output models from this link. Here you will find the example of the plum model under the QNLI stain, without distillation and with adapters for sparsity 0.6, 0.7 and 0.8.

## Distillation 

## "Classic" Fine Tuning

## Adapters
