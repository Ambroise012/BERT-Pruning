# BERT-Pruning
## Paper
This repository contains the code for the report about Structure pruning on BERT : [BERT Pruning](https://ambroise012.github.io/files/BertPruning.pdf)

## Environment

```
conda env create -n prune_llm
conda activate prune_llm
cd BERT-Pruning
```

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
│   ├── output_models
│       └── no_distill
│            └── adapter
│                ├── MNLI
│                ├── SST2
│                ├── ...
│                └── QNLI
│                    ├── 0.6
│                        ├── best
│                        ├── log.txt
│                        ├── pytorch_model.bin
│                        ├── l0_module.pt
│                        ├── config.json
│                        ├── ...
│                        └── zs.pt
│                    ├── 0.7
│                    └── 0.8
|   ├── squad
│       ├── dev-v1.1.json      
│       └── train-v1.1.json                         
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
|   └── README.md                  
```



You can also download the output models from this link. Here you will find the example of the plum model under the QNLI stain, without distillation and with adapters for sparsity 0.6, 0.7 and 0.8.

# Train
## With  Distillation 

```bash
#!/bin/bash

#SBATCH --job-name=instal        # Job name
#SBATCH --output=job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8       # Schedule one core
#SBATCH --gres=gpu:2               # Schedule a GPU
#SBATCH --time=5-00:00:00        #time
#SBATCH --partition=red         #red, brown
#SBATCH --mail-type=END          # Send an email when the job finishes



glue_low=(MRPC RTE STSB CoLA)
glue_high=(MNLI QQP QNLI SST2)

proj_dir=.

code_dir=${proj_dir}

# task and data
task_name=MNLI # for example

# pretrain model
model_name_or_path=bert-base-uncased

# logging & saving
logging_steps=100
save_steps=0


# train parameters
max_seq_length=128
batch_size=32 
learning_rate=2e-5
reg_learning_rate=0.01
epochs=20 

# seed
seed=57

# pruning and distillation
pruning_type=structured_heads+structured_mlp+hidden+layer
distillation_path=google-bert/bert-base-uncased
distill_layer_loss_alpha=0.9
distill_ce_loss_alpha=0.1
distill_temp=2
# 2: fix hidden layers, 3: min distance matching without restriction, 4: min distance matching with restriction
layer_distill_version=4
sparsity_epsilon=0.01

scheduler_type=linear


if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=50
    epochs=100
    start_saving_best_epochs=50
    #prepruning_finetune_epochs=4
    lagrangian_warmup_epochs=20
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    eval_steps=500
    #prepruning_finetune_epochs=1
    lagrangian_warmup_epochs=2
fi

pretrained_pruned_model=None


SPARSITY=(0.6)
for target_sparsity in "${SPARSITY[@]}"; do
    ex_name=${target_sparsity}
    ex_cate=adapter
    output_dir=output_models/${task_name}/${ex_cate}/${ex_name}
    mkdir -p ${output_dir}
    (python3 run_glue_prune.py \
        --output_dir ${output_dir} \
        --logging_steps ${logging_steps} \
        --task_name ${task_name} \
        --model_name_or_path ${model_name_or_path} \
        --ex_cate ${ex_cate} \
        --do_train \
        --do_eval \
        --max_seq_length ${max_seq_length} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --learning_rate ${learning_rate} \
        --reg_learning_rate ${reg_learning_rate} \
        --num_train_epochs ${epochs} \
        --overwrite_output_dir \
        --save_steps ${save_steps} \
        --eval_steps ${eval_steps} \
        --evaluation_strategy steps \
        --seed ${seed} \
        --pruning_type ${pruning_type} \
        --pretrained_pruned_model ${pretrained_pruned_model} \
        --target_sparsity $target_sparsity \
        --freeze_embeddings \
        --do_distill \
        --do_layer_distill \
        --distillation_path $distillation_path \
        --distill_ce_loss_alpha $distill_ce_loss_alpha \
        --distill_loss_alpha $distill_layer_loss_alpha \
        --distill_temp $distill_temp \
        --scheduler_type $scheduler_type \
        --layer_distill_version $layer_distill_version \
        --lagrangian_warmup_epochs $lagrangian_warmup_epochs 2>&1 | tee ${output_dir}/log.txt
        )
done
```

## Without distillation

```bash
#!/bin/bash

#SBATCH --job-name=instal        # Job name
#SBATCH --output=job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=16        # Schedule one core
#SBATCH --gres=gpu:2             # Schedule a GPU
#SBATCH --time=5-00:00:00        #time
#SBATCH --partition=red          #red, brown
#SBATCH --mail-type=END          # Send an email when the job finishes

glue_low=(MRPC RTE STSB CoLA)
glue_high=(MNLI QQP QNLI SST2)

proj_dir=.

code_dir=${proj_dir}

# task and data
task_name=MNLI

# pretrain model
model_name_or_path=bert-base-uncased

# logging & saving
logging_steps=100
save_steps=0

# train parameters
max_seq_length=128
batch_size=32 
learning_rate=2e-5
reg_learning_rate=0.01
epochs=20 

# seed
seed=57

# pruning
pruning_type=structured_heads+structured_mlp+hidden+layer
#distillation_path=google-bert/bert-base-uncased
#distill_layer_loss_alpha=0.9
#distill_ce_loss_alpha=0.1
#distill_temp=2
# 2: fix hidden layers, 3: min distance matching without restriction, 4: min distance matching with restriction
#layer_distill_version=4
sparsity_epsilon=0.01

scheduler_type=linear

if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=50
    epochs=100
    start_saving_best_epochs=50
    #prepruning_finetune_epochs=4
    lagrangian_warmup_epochs=20
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    eval_steps=500
    #prepruning_finetune_epochs=1
    lagrangian_warmup_epochs=2
fi

pretrained_pruned_model=output_models/MRPC/no_distill/0.8

target_sparsity=0.8
ex_name=${target_sparsity}
ex_cate=no_distill/adapter
output_dir=output_all_model
mkdir -p ${output_dir}
(python3 run_glue_prune.py \
    --output_dir ${output_dir} \
    --logging_steps ${logging_steps} \
    --task_name ${task_name} \
    --model_name_or_path ${model_name_or_path} \
    --ex_cate ${ex_cate} \
    --do_train \
    --do_eval \
    --max_seq_length ${max_seq_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 32 \
    --learning_rate ${learning_rate} \
    --reg_learning_rate ${reg_learning_rate} \
    --num_train_epochs ${epochs} \
    --overwrite_output_dir \
    --save_steps ${save_steps} \
    --eval_steps ${eval_steps} \
    --evaluation_strategy steps \
    --seed ${seed} \
    --pruning_type ${pruning_type} \
    --pretrained_pruned_model ${pretrained_pruned_model} \
    --target_sparsity $target_sparsity \
    --freeze_embeddings \
    --scheduler_type $scheduler_type \
    --lagrangian_warmup_epochs $lagrangian_warmup_epochs 2>&1 | tee ${output_dir}/log.txt
    )
```

## Adapters
Without distillation : copy the train code without distillation, and run the script "run_adapter.py".

## Evaluation
Example for the task QNLI :

```bash
category=no_distill # train or fine_tune or no_distill
SPARSITY=(0.6 0.7 0.8 0.9 0.95)
for sparsity_target in "${SPARSITY[@]}"; do 
    python3 evaluation.py QNLI output_models/QNLI/${category}/${sparsity_target}/best
done
```
Save the following results in csv file : task name, model size, sparsity, accuracy, speedup (seconds), emissions

