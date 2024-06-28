import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import csv
import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.trainer_utils import (PREFIX_CHECKPOINT_DIR, EvalPrediction,
                                        EvaluationStrategy, PredictionOutput,
                                        TrainOutput)
from transformers.utils import logging
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from args import AdditionalArguments
from utils.cofi_utils import *
from utils.utils import *
from torch.utils.data import (RandomSampler, SequentialSampler,
                              TensorDataset)
import wandb
from utils.utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)
import argparse
import random
import glob
from datasets import load_dataset, load_metric, DatasetDict
from transformers import AutoConfig, AutoTokenizer, default_data_collator, DataCollatorWithPadding
from transformers import (HfArgumentParser, PretrainedConfig,
                          glue_output_modes, glue_tasks_num_labels, set_seed)
from args import DataTrainingArguments
from models.l0_module import L0Module
from models.modeling_bert import CoFiBertForSequenceClassification
from models.modeling_roberta import CoFiRobertaForSequenceClassification
from trainer.trainer import CoFiTrainer
from models.model_args import ModelArguments
from transformers.trainer import Trainer
from codecarbon import EmissionsTracker
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForQuestionAnswering, BertTokenizer,
                          XLMConfig, XLMForQuestionAnswering,
                          XLMTokenizer, XLNetConfig,
                          XLNetForQuestionAnswering,
                          XLNetTokenizer)
from utils.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad
from models.l0_module import *
logger = logging.get_logger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer)
}

class SquadTrainer:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        self.tokenizer = self.tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        self.model = self.model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=self.config)
        self.model.to(self.device)

        self.l0_module = None
        self.prepruning_finetune_steps = 100
        self.start_prune = False
        self.optimizer = None  
        self.lr_scheduler = None
        self.total_flos = 0

        self.l0_optimizer = None
        self.lagrangian_optimizer = None

        self.floating_point_ops

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0
################   ajout de la funct create_optimizer_and_scheduler   ###############################
    def create_optimizer_and_scheduler(self, num_training_steps: int, build_l0_optimizer:bool=True):
        def log_params(param_groups, des):
            for i, grouped_parameters in enumerate(param_groups):
                logger.info(
                    f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, weight_decay: {grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")

        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            freeze_keywords = ["embeddings"]

            main_model_params = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                },
            ]
            log_params(main_model_params, "main params")
            self.optimizer = AdamW(
                main_model_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

            if build_l0_optimizer is not None:
                l0_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" not in n],
                    "weight_decay": 0.0,
                    "lr": self.args.reg_learning_rate
                }]
                log_params(l0_params, "l0 reg params")
                self.l0_optimizer = AdamW(l0_params,
                                          betas=(self.args.adam_beta1,
                                                 self.args.adam_beta2),
                                          eps=self.args.adam_epsilon, )

                lagrangian_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" in n],
                    "weight_decay": 0.0,
                    "lr": -self.args.reg_learning_rate
                }]
                log_params(lagrangian_params, "l0 reg lagrangian params")
                self.lagrangian_optimizer = AdamW(lagrangian_params,
                                                    betas=(self.args.adam_beta1,
                                                            self.args.adam_beta2),
                                                    eps=self.args.adam_epsilon)

        if self.lr_scheduler is None:
            if self.args.scheduler_type == "linear":
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
                )
            else:
                self.lr_scheduler = None
######################################################################################################
    def _setup_device(self):
        if self.args.local_rank == -1 or self.args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
            self.args.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.args.n_gpu = 1
        return device

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)

    def to_list(self, tensor):
        return tensor.detach().cpu().tolist()
    def train(self, train_dataset):
        self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        train_sampler = RandomSampler(train_dataset) if self.args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        if self.l0_module is not None:
            lagrangian_warmup_steps = self.args.lagrangian_warmup_epochs * num_update_steps_per_epoch
            self.prepruning_finetune_steps = self.args.prepruning_finetune_epochs * num_update_steps_per_epoch
            self.l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)
            logger.info(f"Prepruning finetune steps: {self.prepruning_finetune_steps}")
            logger.info(f"Lagrangian warmup steps: {lagrangian_warmup_steps}")

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total, build_l0_optimizer=self.start_prune)

        model = self.model

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", self.args.train_batch_size * self.args.gradient_accumulation_steps * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        # if self.l0_module is not None:
        self.l0_module.zero_grad()

        self.optimizer.zero_grad()
        if self.l0_optimizer is not None:
            self.l0_optimizer.zero_grad()
        if self.lagrangian_optimizer is not None:
            self.lagrangian_optimizer.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch", disable=self.args.local_rank not in [-1, 0])
        self.set_seed()

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': None if self.args.model_type == 'xlm' else batch[2], 'start_positions': batch[3], 'end_positions': batch[4]}
                if self.args.model_type in ['xlnet', 'xlm']:
                    inputs.update({'cls_index': batch[5], 'p_mask': batch[6]})
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.n_gpu > 1:
                    loss = loss.mean()
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    scheduler.step()
                    optimizer.step()
                    self.model.zero_grad()
                    global_step += 1

                    # if self.args.local_rank in [-1, 0] and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    #     output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(global_step))
                    #     if not os.path.exists(output_dir):
                    #         os.makedirs(output_dir)
                    #     model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    #     model_to_save.save_pretrained(output_dir)
                    #     torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
                    #     logger.info("Saving model checkpoint to %s", output_dir)

                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and global_step > self.args.max_steps:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

#     def train(self, train_dataset):
#         self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
#         train_sampler = RandomSampler(train_dataset) if self.args.local_rank == -1 else DistributedSampler(train_dataset)
#         train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
# ####################  ajout  #############################
#         num_update_steps_per_epoch = len(
#         train_dataloader) // self.args.gradient_accumulation_steps
#         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1) #! 12272

#         if self.l0_module is not None:
#             lagrangian_warmup_steps = self.args.lagrangian_warmup_epochs * num_update_steps_per_epoch #! 24544
#             self.prepruning_finetune_steps = self.args.prepruning_finetune_epochs * num_update_steps_per_epoch
#             self.l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)
#             logger.info(f"Prepruning finetune steps: {self.prepruning_finetune_steps}")
#             logger.info(f"Lagrangian warmup steps: {lagrangian_warmup_steps}")

#         # if self.args.max_steps > 0:
#         #     self.t_total = self.args.max_steps
#         #     num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
#         #         self.args.max_steps % num_update_steps_per_epoch > 0
#         #     )
#         # else:
#         #     self.t_total = int(num_update_steps_per_epoch *
#         #                        self.args.num_train_epochs)
#         #     num_train_epochs = self.args.num_train_epochs
#         #     self.args.max_steps = self.t_total
# #############################################################
#         if self.args.max_steps > 0:
#             t_total = self.args.max_steps
#             self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
#         else:
#             t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

# ##############################################################
#         self.create_optimizer_and_scheduler(num_training_steps=t_total, build_l0_optimizer = self.start_prune)

#         model = self.model

# ###############################################################          
#         no_decay = ['bias', 'LayerNorm.weight']
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
#             {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]
#         optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
#         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

#         if self.args.n_gpu > 1:
#             self.model = torch.nn.DataParallel(self.model)

#         if self.args.local_rank != -1:
#             self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True)

#         logger.info("***** Running training *****")
#         logger.info("  Num examples = %d", len(train_dataset))
#         logger.info("  Num Epochs = %d", self.args.num_train_epochs)
#         logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
#         logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", self.args.train_batch_size * self.args.gradient_accumulation_steps * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1))
#         logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
#         logger.info("  Total optimization steps = %d", t_total)

#         global_step = 0
#         tr_loss, logging_loss = 0.0, 0.0
#         self.model.zero_grad()
#         ##################################
#         if self.l0_module is not None:
#             self.l0_module.zero_grad()
        
#         self.optimizer.zero_grad()
#         if self.l0_optimizer is not None:
#             self.l0_optimizer.zero_grad()
#         if self.lagrangian_optimizer is not None:
#             self.lagrangian_optimizer.zero_grad()
#         ###################################
#         train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch", disable=self.args.local_rank not in [-1, 0])
#         self.set_seed()

#         for _ in train_iterator:
#             epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0])
#             for step, batch in enumerate(epoch_iterator):
#                 ################################################
#                 if self.prepruning_finetune_steps > 0 and global_step == self.prepruning_finetune_steps: #! before pruning, run 12272 steps
#                     self.start_prune = True

#                     self.optimizer = None
#                     self.lr_scheduler = None
#                     lr_steps = t_total - global_step

#                     # reset the optimizer
#                     self.create_optimizer_and_scheduler(lr_steps, self.start_prune)
#                     logger.info("Starting l0 regularization!")

#                 if self.start_prune and self.l0_module is not None:
#                     zs = self.l0_module.forward(training=True) #! get the zs
#                     self.fill_inputs_with_zs(zs, inputs) #! use the zs

#                 # loss_terms = self.training_step(model, inputs)
#                 # tr_loss_step = loss_terms["loss"]
#                 # lag_loss_step = loss_terms["lagrangian_loss"]

#                 # tr_loss += tr_loss_step
#                 # lag_loss += lag_loss_step if lag_loss_step is not None else 0.0

#                 self.total_flos += self.floating_point_ops(inputs)    

#                 if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
#                         len(epoch_iterator) <= self.args.gradient_accumulation_steps
#                         and (step + 1) == len(epoch_iterator)
#                 ):
#                     torch.nn.utils.clip_grad_norm_(
#                         model.parameters(), self.args.max_grad_norm)

#                     self.optimizer.step()

#                     if self.l0_module is not None and self.l0_optimizer is not None:
#                         self.l0_optimizer.step()
#                         self.lagrangian_optimizer.step()

#                     if self.lr_scheduler is not None:
#                         self.lr_scheduler.step()

#                     if self.l0_module is not None:
#                         self.l0_module.constrain_parameters()

#                     model.zero_grad()
#                     if self.l0_module is not None:
#                         self.l0_module.zero_grad()
#                     self.optimizer.zero_grad()
#                     if self.l0_optimizer is not None:
#                         self.l0_optimizer.zero_grad()
#                     if self.lagrangian_optimizer is not None:
#                         self.lagrangian_optimizer.zero_grad()
#                 #################################################
#                 self.model.train()
#                 batch = tuple(t.to(self.args.device) for t in batch)
#                 inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': None if self.args.model_type == 'xlm' else batch[2], 'start_positions': batch[3], 'end_positions': batch[4]}
#                 if self.args.model_type in ['xlnet', 'xlm']:
#                     inputs.update({'cls_index': batch[5], 'p_mask': batch[6]})
#                 outputs = self.model(**inputs)
#                 loss = outputs[0]

#                 if self.args.n_gpu > 1:
#                     loss = loss.mean()
#                 if self.args.gradient_accumulation_steps > 1:
#                     loss = loss / self.args.gradient_accumulation_steps

#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

#                 tr_loss += loss.item()
#                 if (step + 1) % self.args.gradient_accumulation_steps == 0:
#                     scheduler.step()
#                     optimizer.step()
#                     self.model.zero_grad()
#                     global_step += 1

#                     if self.args.local_rank in [-1, 0] and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
#                         output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(global_step))
#                         if not os.path.exists(output_dir):
#                             os.makedirs(output_dir)
#                         model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
#                         model_to_save.save_pretrained(output_dir)
#                         torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
#                         logger.info("Saving model checkpoint to %s", output_dir)

#                 if self.args.max_steps > 0 and global_step > self.args.max_steps:
#                     epoch_iterator.close()
#                     break
#             if self.args.max_steps > 0 and global_step > self.args.max_steps:
#                 train_iterator.close()
#                 break

#         return global_step, tr_loss / global_step
    

    # def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
    #     model.train()
    #     if self.l0_module is not None:
    #         self.l0_module.train()
    #     inputs = self._prepare_inputs(inputs)

    #     distill_loss = None
    #     distill_ce_loss = None
    #     if self.teacher_model is not None:
    #         with torch.no_grad():
    #             # only retain inputs of certain keys
    #             teacher_inputs_keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids", "labels",
    #                                    "output_attentions", "output_hidden_states", "return_dict"]
    #             teacher_inputs = {key: inputs[key]
    #                               for key in teacher_inputs_keys if key in inputs}
    #             self.shortens_inputs(teacher_inputs)
    #             teacher_outputs = self.teacher_model(**teacher_inputs)
    #         self.shortens_inputs(inputs)
    #         student_outputs = model(**inputs) #! get the two outputs

    #         zs = {key: inputs[key] for key in inputs if "_z" in key} #! extract the zs
    #     else:
    #         loss = self.compute_loss(model, inputs)

    #     lagrangian_loss = None
    #     if self.start_prune and self.l0_module is not None:
    #         lagrangian_loss, _, _ = \
    #             self.l0_module.lagrangian_regularization(
    #                 self.global_step - self.prepruning_finetune_steps)
    #         loss += lagrangian_loss

    #     if self.args.gradient_accumulation_steps > 1:
    #         loss = loss / self.args.gradient_accumulation_steps

    #     loss.backward()
    #     # with open('loss.csv', 'a', newline='') as csvfile:
    #     #     writer = csv.writer(csvfile)
    #     #     writer.writerow([loss.item(), 
    #     #              lagrangian_loss.item() if lagrangian_loss is not None else None, 
    #     #              distill_loss.item() if distill_loss is not None else None, 
    #     #              distill_ce_loss.item() if distill_ce_loss is not None else None])

    #     return {"loss": loss.detach(),
    #             "lagrangian_loss": lagrangian_loss.detach() if lagrangian_loss is not None else None,
    #             "distill_layer_loss": distill_loss.detach() if distill_loss is not None else None,
    #             "distill_ce_loss": distill_ce_loss.detach() if distill_ce_loss is not None else None}
    def calculate_parameters(module):
        keys = ["embedding", "layer_transformation", "classifier", "pooler"]
        return sum(p.numel() for n, p in module.named_parameters() if not any(key in n for key in keys))
    
    def evaluate(self, model, tokenizer, prefix=""):
        dataset, examples, features = self.load_and_cache_examples(tokenizer, evaluate=True, output_examples=True)

        if not os.path.exists(self.args.output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.args.output_dir)

        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        eval_sampler = SequentialSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        all_results = []

        start_time = time.time() 
        tracker = EmissionsTracker()
        tracker.start()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': None if self.args.model_type == 'xlm' else batch[2]}
                example_indices = batch[3]
                if self.args.model_type in ['xlnet', 'xlm']:
                    inputs.update({'cls_index': batch[4], 'p_mask': batch[5]})
                outputs = self.model(**inputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                if self.args.model_type in ['xlnet', 'xlm']:
                    result = RawResultExtended(unique_id=unique_id, start_top_log_probs=self.to_list(outputs[0][i]), start_top_index=self.to_list(outputs[1][i]), end_top_log_probs=self.to_list(outputs[2][i]), end_top_index=self.to_list(outputs[3][i]), cls_logits=self.to_list(outputs[4][i]))
                else:
                    result = RawResult(unique_id=unique_id, start_logits=self.to_list(outputs[0][i]), end_logits=self.to_list(outputs[1][i]))
                all_results.append(result)

        inference_time = time.time() - start_time
        output_prediction_file = os.path.join(self.args.output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(self.args.output_dir, "nbest_predictions_{}.json".format(prefix))
        if self.args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.args.output_dir, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None

        if self.args.model_type in ['xlnet', 'xlm']:
            write_predictions_extended(examples, features, all_results, self.args.n_best_size,
                            self.args.max_answer_length, output_prediction_file,
                            output_nbest_file, output_null_log_odds_file, self.args.predict_file,
                            model.config.start_n_top, model.config.end_n_top,
                            self.args.version_2_with_negative, tokenizer, self.args.verbose_logging)
        else:
            write_predictions(examples, features, all_results, self.args.n_best_size,
                            self.args.max_answer_length, self.args.do_lower_case, output_prediction_file,
                            output_nbest_file, output_null_log_odds_file, self.args.verbose_logging,
                            self.args.version_2_with_negative, self.args.null_score_diff_threshold)
        # Evaluate with the official SQuAD script
        evaluate_options = EVAL_OPTS(data_file=self.args.predict_file,
                                    pred_file=output_prediction_file,
                                    na_prob_file=output_null_log_odds_file)
        emissions: float = tracker.stop()

        results = evaluate_on_squad(evaluate_options)
        results['inference_time'] = inference_time
        model_size = sum(p.numel() for p in model.parameters())
        results['model_size'] = model_size
        results["emissions"] = emissions
        return results
    
    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0
    def fill_inputs_with_zs(self, zs, inputs):
        for key in zs:
            inputs[key] = zs[key]
    def prediction_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None) -> PredictionOutput:
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # disable output hidden states and attention during evaluation
        self.config.output_hidden_states = False
        self.config.output_attentions = False

        model = self.model

        # multi-gpu eval
        model = self.model

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm

        zs = None
        if self.start_prune and self.l0_module is not None:
            self.l0_module.eval()
            zs = self.l0_module.forward(training=False)

        if zs is not None:
            pruned_model_size_info = self.l0_module.calculate_model_size(zs)

        for ii, inputs in enumerate(tqdm(dataloader, desc=description, disable=disable_tqdm)):
            if zs is not None:
                if ii == 0:
                    logger.info(f"Putting zs {zs.keys()} into inputs:")
                self.fill_inputs_with_zs(zs, inputs) #! use the zs
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only)

            batch_size = inputs[list(inputs.keys())[0]].shape[0]

            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels)
            if loss is not None:
                if type(loss) == float:
                    losses = [loss] * batch_size
                    if losses_host is None:
                        losses_host = losses
                    else:
                        losses_host.extend(losses)
                else:
                    losses = loss.repeat(batch_size)
                    losses_host = losses if losses_host is None else torch.cat(
                        (losses_host, losses), dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation.py loop
            delattr(self, "_past")

        if losses_host is not None:
            if not torch.is_tensor(losses_host):
                losses_host = torch.tensor(losses_host)
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(
                predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        if all_losses is not None and len(all_losses) > 0:
            metrics["eval_loss"] = np.mean(all_losses)

        if zs is not None:
            lag_loss, expected_sparsity, target_sparsity = self.l0_module.lagrangian_regularization(
                self.global_step - self.prepruning_finetune_steps)

            expected_sparsity = round(expected_sparsity.item(), 5)
            metrics.update(pruned_model_size_info)
            metrics["expected_sparsity"] = expected_sparsity
            metrics["target_sparsity"] = target_sparsity

            if (not self.start_saving_best) and (expected_sparsity - self.additional_args.target_sparsity >= -self.additional_args.sparsity_epsilon):
                self.start_saving_best = True
                logger.info(f"Starting saving the best from epoch {int(self.epoch)} and step {self.global_step}")

        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True

        return PredictionOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics)
    
    def load_and_cache_examples(self, tokenizer, evaluate=False, output_examples=False):
        # Load data features from cache or dataset file
        input_file = self.args.predict_file if evaluate else self.args.train_file
        cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, self.args.model_name_or_path.split('/'))).pop(),
            str(self.args.max_seq_length)))

        if os.path.exists(cached_features_file) and not self.args.overwrite_cache and not output_examples:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
            
        else:
            logger.info("Creating features from dataset file at %s", input_file)
            examples = read_squad_examples(input_file=input_file,
                                        is_training=not evaluate,
                                        version_2_with_negative=self.args.version_2_with_negative)
            features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=self.args.max_seq_length,
                                                doc_stride=self.args.doc_stride,
                                                max_query_length=self.args.max_query_length,
                                                is_training=not evaluate)
        if self.args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        if evaluate:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_example_index, all_cls_index, all_p_mask)
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)

        if output_examples:
            return dataset, examples, features
        return dataset
