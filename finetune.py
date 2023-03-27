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
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"


class DatasetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.dataset[idx]["input_ids"])[:-1],
            torch.LongTensor(self.dataset[idx]["input_ids"])[1:],
        )


class RepeatingLoader:
    def __init__(self, loader):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (iterator): The data loader to repeat.
        """
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch


def model_forward(model, inputs):
    h = inputs
    h = h.to(model.base_model.model.model.embed_tokens.weight.device)
    h = model.base_model.model.model.embed_tokens(h)
    for layer in model.base_model.model.model.layers:
        h = h.to(layer.input_layernorm.weight.device)
        h = layer(h)[0]
    h = h.to(model.base_model.model.model.norm.weight.device)
    h = model.base_model.model.model.norm(h)
    h = model.base_model.model.lm_head(h)
    return h


class BetterTrainer(transformers.Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        if not training:
            return model
        if self.args.torch_compile:
            model = torch.compile(
                model, backend=self.args.torch_compile_backend, mode=self.args.torch_compile_mode)

        return model

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        logits = model_forward(model, inputs['input_ids'])
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1,
                                  model.config.vocab_size).to(labels.device),
                shift_labels.view(-1)
            )

        return (loss, logits) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "./alpaca_data_cleaned.json",
    output_dir: str = "./lora-alpaca",
    dataset_uri: str = "./aerner-guanaco-v1-512",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
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

    save_total_limit: int = 4,
    num_train_steps=20000,

    warmup_steps: int = 100,
    logging_steps: int = 10,
    eval_steps: int = 200,
    save_steps: int = 200,

    group_by_length: bool = False,  # faster, but produces an odd training loss curve,
    # either training checkpoint or final adapter
    resume_from_checkpoint: str = None,
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
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
    # Dataset
    #
    #
    # data = load_dataset("json", data_files=data_path)
    data = datasets.load_dataset("JosephusCheung/GuanacoDataset")
    dataset = data["train"].select(range(10000)).map(
        generate_and_tokenize_prompt)
    dataset = dataset.remove_columns(["instruction", "input", "output"])
    dataset = dataset.train_test_split(test_size=val_set_size, seed=0)

    #
    #
    #
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    #
    # device_map
    #
    config = AutoConfig.from_pretrained(base_model)
    device_ids = list(range(torch.cuda.device_count()))
    device_map = {
        "model.embed_tokens": device_ids[0],
        "model.norm.weight": device_ids[-1],
        "lm_head": device_ids[-1],
    }
    allocations = [
        device_ids[i] for i in
        sorted(list(range(len(device_ids))) *
               math.ceil(config.num_hidden_layers / len(device_ids)))
    ]
    for layer_i, device_id in enumerate(allocations):
        device_map[f"model.layers.{layer_i}"] = device_id

    print(device_ids)
    print(device_map)

    #
    #
    # Model
    #
    #
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head.to(torch.float16)
    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False
    model = prepare_model_for_int8_training(model)

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
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    #
    # Initial save
    #
    model.save_pretrained(output_dir)
    print("Saved initial model")

    #
    #
    # Train loop
    #
    #

    # dataloader = RepeatingLoader(torch.utils.data.DataLoader(
    #     DatasetDataset(data["train"]),
    #     batch_size=micro_batch_size,
    #     shuffle=True
    # ))

    # print("Setup optimizer")
    # opt = torch.optim.AdamW([
    #     p
    #     for p in model.parameters()
    #     if p.requires_grad
    # ], lr=learning_rate)

    # # Train (maybe can replace with Trainer? I think Trainer might mess up the device mappings though.)
    # print("Start training")
    # generator = iter(dataloader)
    # for step in tqdm.trange(num_train_steps, initial=0):
    #     input_ids, labels = next(generator)
    #     logits = model_forward(model, input_ids)
    #     loss = torch.nn.functional.cross_entropy(
    #         logits.view(-1, model.config.vocab_size),
    #         labels.view(-1).to(logits.device),
    #     )
    #     loss.backward()
    #     opt.step()

    #     actual_step = step + 1

    #     if step % 10 == 0:
    #         print(f"Loss={loss.item():.3f}")

    #     if actual_step % gradient_accumulation_steps == 0:
    #         opt.zero_grad()

    #     if actual_step % save_steps == 0:
    #         model.save_pretrained(output_dir)

    #
    #
    # Trainer
    #
    #
    trainer = BetterTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            report_to="wandb",

            fp16=True,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,

            # per_device_train_batch_size=128,
            # per_device_eval_batch_size=128,
            # auto_find_batch_size=True,
            per_device_train_batch_size=micro_batch_size,
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
