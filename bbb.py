
import transformers
import peft
import datasets
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PeftArguments:
    peft_type: str
    r: dataclass[int]
    lora_alpha: dataclass[int]
    lora_dropout: dataclass[float]
    # Used for prompt tuning, prefix tuning and p-tuning
    num_virtual_tokens: dataclass[int]
    mapping_hidden_dim: dataclass[int]

# @dataclass
# class PeftArguments(peft.LoraConfig, peft.PromptTuningConfig):
#     pass

# a_args = transformers.HfArgumentParser((peft.PromptTuningConfig)).parse_args_into_dataclasses()
arg_parser = transformers.HfArgumentParser((PeftArguments))
aaa = arg_parser.parse_args_into_dataclasses()

print(aaa)
