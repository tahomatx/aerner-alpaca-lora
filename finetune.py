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
from datasets import load_dataset
import transformers
import datasets
import math

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
    cutoff_len: int = 512,
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
    group_by_length: bool = True,  # faster, but produces an odd training loss curve
    save_total_limit: int = 4,

    num_train_steps=20000,
    save_steps=200,
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
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

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
    #
    # Dataset
    #
    #
    # dataset = datasets.load_from_disk(dataset_uri).train_test_split(
    #     test_size=val_set_size, seed=0)
    # dataset = dataset.remove_columns(["instruction", "input", "output"])

    data = load_dataset("json", data_files=data_path)
    data = data.map(generate_and_tokenize_prompt)
    data = data.remove_columns(["instruction", "input", "output"])
    dataset = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=0)

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

    class BetterTrainingArguments(transformers.TrainingArguments):
        @property
        def place_model_on_device(self):
            return False

    class BetterTrainer(transformers.Trainer):
        def _wrap_model(self, model, training=True, dataloader=None):
            # already initialized its own DDP and AMP
            if self.deepspeed:
                return self.deepspeed

            # Note: in torch.distributed mode, there's no point in wrapping the model
            # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
            if not training:
                return model

            # torch.compile() needs to be called after wrapping the model with FSDP or DDP
            # to ensure that it accounts for the graph breaks required by those wrappers
            if self.args.torch_compile:
                model = torch.compile(model, backend=self.args.torch_compile_backend, mode=self.args.torch_compile_mode)

            return model

        def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """
            # if self.label_smoother is not None and "labels" in inputs:
            #     labels = inputs.pop("labels")
            # else:
            #     labels = None

            print(inputs)
            input_ids, labels = inputs
            outputs = model_forward(model, input_ids)
            print(outputs)
            # outputs = model(**inputs.to("cuda:0"))
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss


        def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
            """
            Perform a training step on a batch of inputs.
            Subclass and override to inject custom behavior.
            Args:
                model (`nn.Module`):
                    The model to train.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.
                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
            Return:
                `torch.Tensor`: The tensor with training loss on this batch.
            """
            model.train()
            inputs = self._prepare_inputs(inputs)

            # if is_sagemaker_mp_enabled():
            #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            #     return loss_mb.reduce_mean().detach().to(self.args.device)

            # with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

            # if self.args.n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            #     # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            #     loss = loss / self.args.gradient_accumulation_steps

            # if self.do_grad_scaling:
            #     self.scaler.scale(loss).backward()
            # elif self.use_apex:
            #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # elif self.deepspeed:
            #     # loss gets scaled under gradient_accumulation_steps in deepspeed
            #     loss = self.deepspeed.backward(loss)
            # else:
            loss.backward()

            return loss.detach()


    trainer = BetterTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=BetterTrainingArguments(
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
            logging_steps=10,

            evaluation_strategy="steps" if val_set_size > 0 else "no",
            eval_steps=200 if val_set_size > 0 else None,

            save_strategy="steps",
            save_steps=200,
            save_total_limit=save_total_limit,

            warmup_steps=100,
            learning_rate=learning_rate,  # the Karpathy constant
            # group_by_length=group_by_length,
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

    print("Starting training")
    trainer.train()

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
