"""Pretraining Llama on Malayalam data.
References:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
https://huggingface.co/docs/peft/en/task_guides/token-classification-lora
"""

import logging
import numpy as np
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import datasets
import torch
from datasets import load_dataset, concatenate_datasets


import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import get_last_checkpoint, PREFIX_CHECKPOINT_DIR
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from sklearn.metrics import accuracy_score
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    get_peft_model_state_dict,
)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "pt_lora_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        peft_model_path = os.path.join(args.output_dir, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)


def accuracy(prediction, references, normalize=True, sample_weight=None):
    return {
        "accuracy": float(
            accuracy_score(
                references, prediction, normalize=normalize, sample_weight=sample_weight
            )
        )
    }


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(preds, labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # depending on the model and config, logits may contain extra tensors.
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # special handling for labels.
    # ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features])
        
        # handling of all other possible keys.
        # again we will use the first element to figure out which key/values are not None for this model.
        try:
            for k, v in first.items():
                if k not in ("label", 'label_ids') and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
        except ValueError:
            for k, v in first.items():
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([features[0][k]] * len(features))
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                    else:
                        batch[k] = torch.tensor([features[0][k]] * len(features))
    return batch
