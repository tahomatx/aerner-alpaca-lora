# pip uninstall -y transformers && pip install -q git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/peft.git bitsandbytes datasets accelerate sentencepiece wandb fire


from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer
import tqdm.auto as tqdm
import os
import sys
from typing import List

import fire
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
import datasets
import math
from torch.nn import CrossEntropyLoss

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    output_dir: str = "./lora-alpaca",
    dataset_uri: str = "JosephusCheung/GuanacoDataset",

    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    learning_rate: float = 3e-4,

    # training hyperparams
    batch_size: int = 256,
    micro_batch_size: int = 64,
    cutoff_len: int = 512,

    dataset_size=65536,
    val_set_size: int = 1024,

    num_epochs: int = 3,
    warmup_steps: int = 128,
    logging_steps: int = 1,
    eval_steps: int = 32,
    save_steps: int = 32,
    save_total_limit: int = 16,

    group_by_length: bool = False,  # faster, but produces an odd training loss curve,
    # either training checkpoint or final adapter
    resume_from_checkpoint: str = None,
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"dataset_uri: {dataset_uri}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    #
    #
    # Tokenizer
    #
    #
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=True)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    #
    #
    # Dataset
    #
    #
    # data = load_dataset("json", data_files=data_path)
    # data = datasets.load_dataset(dataset_uri)
    dataset = datasets.load_from_disk("./aerner-guanaco-ja")
    dataset = data["train"].select(range(dataset_size + val_set_size)).map(
        generate_and_tokenize_prompt)
    dataset = dataset.remove_columns(["instruction", "input", "output"])
    dataset = dataset.train_test_split(test_size=val_set_size, seed=0)

    # Data check
    for d in dataset["train"].select(range(10)):
        print(d['input_ids'])
        print("")

    #
    #
    # Model
    #
    #
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map='auto',
    )

    model = prepare_model_for_int8_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False

    #
    #
    # Apply LoRA
    #
    #
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    #
    # Initial save
    #
    model.save_pretrained(output_dir)
    print("Saved initial model")

    #
    #
    # Trainer
    #
    #
    print(len(dataset["train"]), len(dataset["test"]))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            report_to="wandb",

            fp16=True,
            load_best_model_at_end=True if val_set_size > 0 else False,

            # auto_find_batch_size=True,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,

            num_train_epochs=num_epochs,

            logging_strategy="steps",
            logging_steps=logging_steps,

            evaluation_strategy="steps" if val_set_size > 0 else "no",
            eval_steps=eval_steps if val_set_size > 0 else None,

            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,

            warmup_steps=warmup_steps,
            learning_rate=learning_rate,  # the Karpathy constant
            # group_by_length=group_by_length,

            label_smoothing_factor=0.1,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *
        _, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
