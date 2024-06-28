import logging
import os
import sys
import time
import random
from copy import deepcopy
import argparse
import datasets
import numpy as np
import glob
import torch
import transformers
from datasets import load_dataset, load_metric, DatasetDict
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, default_data_collator, DataCollatorWithPadding
from transformers import (HfArgumentParser, TrainingArguments, PretrainedConfig,
                          glue_output_modes, glue_tasks_num_labels, set_seed)

from args import AdditionalArguments, DataTrainingArguments
from utils.cofi_utils import *
from models.l0_module import L0Module
from models.modeling_bert import CoFiBertForSequenceClassification
from models.modeling_roberta import CoFiRobertaForSequenceClassification
from trainer.trainer_squad import  SquadTrainer
from utils.utils import *
from models.model_args import ModelArguments
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer)
from utils.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad
from codecarbon import EmissionsTracker
import csv

import wandb

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
}
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--predict_file", default=None, type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--version_2_with_negative", action='store_true')
    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)
    parser.add_argument("--max_seq_length", default=384, type=int)
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--evaluate_during_training", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--overwrite_output_dir", action='store_true')
    parser.add_argument("--overwrite_cache", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    parser.add_argument("--n_best_size", default=20, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--verbose_logging", action='store_true')
    parser.add_argument("--lang_id", default=0, type=int)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--eval_all_checkpoints", action='store_true')
    ######## pruning ##########
    parser.add_argument("--pruning_type", default="structured_heads+structured_mlp+hidden+layer",type=str)
    parser.add_argument("--lagrangian_warmup_epochs", default=2, type=int)
    parser.add_argument("--temperature", type=float, default=2./3.)
    parser.add_argument("--prepruning_finetune_epochs", type=int, default=1)
    parser.add_argument("--reg_learning_rate", type=float, default=0.1)
    parser.add_argument("--scheduler_type", type=str, default="linear")
    parser.add_argument("--droprate_init", type=float, default=0.5)
    parser.add_argument("--target_sparsity", type=float, default=0)
    parser.add_argument("--model_revision", type=str, default="main")
    parser.add_argument("--use_auth_token", type=bool, default=False)
##############
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 for Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for Adam optimizer")


    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
     # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)
    Model = CoFiBertForSequenceClassification if args.model_name_or_path.startswith(
        "bert") else CoFiRobertaForSequenceClassification
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    # config = config_class.from_pretrained(
    #     args.model_name_or_path,
    #     num_labels=2,
    #     finetuning_task= args.train_file,
    #     cache_dir=args.cache_dir,
    #     revision=args.model_revision,
    #     use_auth_token=True if args.use_auth_token else None,
    # )
    # config = AutoConfig.from_pretrained(
    #     'google-bert/bert-base-uncased',
    #     cache_dir='/path/to/custom/cache'
    # )
    # if args.pretrained_pruned_model is not None:
    #     zs = load_zs(args.pretrained_pruned_model)
    #     model = load_model(args.pretrained_pruned_model, Model, zs)
    #     print(f"Model Size after pruning: {calculate_parameters(model)}")
    l0_module = None
    if args.pruning_type is not None:
        l0_module = L0Module(config=config,
                             droprate_init=args.droprate_init,
                             temperature=args.temperature,
                             target_sparsity=args.target_sparsity,
                             pruning_type=args.pruning_type)
        
        # model=model,
        # args=training_args,
        # additional_args=additional_args,
        # train_dataset=train_dataset if training_args.do_train else None,
        # eval_dataset=eval_dataset if training_args.do_eval else None,
        # compute_metrics=compute_metrics,
        # tokenizer=tokenizer,
        # data_collator=data_collator,
        # l0_module=l0_module,
        # teacher_model=teacher_model
   
   
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # args.model_type = args.model_type.lower()
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    
    trainer = SquadTrainer(args)
    
    logger.info("Training/evaluation parameters %s", args)
    print("#########   TRAIN   ##########")
    ########### TRAIN ################
    if args.do_train:
        train_dataset = trainer.load_and_cache_examples(tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = trainer.train(train_dataset)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    results = {}
    if args.do_eval:
        results = trainer.evaluate(model,tokenizer)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            # checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            all_checkpoints = sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True))
            checkpoints = [os.path.dirname(c) for idx, c in enumerate(all_checkpoints) if idx % 50 == 0]
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = trainer.evaluate(model, tokenizer)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))
    print(results)

    return results

if __name__ == "__main__":
    main()
