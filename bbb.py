
import transformers
import peft
import datasets
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PeftArguments:
    peft_type: str
    r: Optional[int]
    lora_alpha: Optional[int]
    lora_dropout: Optional[float]
    # Used for prompt tuning, prefix tuning and p-tuning
    num_virtual_tokens: Optional[int]
    mapping_hidden_dim: Optional[int]

# @dataclass
# class PeftArguments(peft.LoraConfig, peft.PromptTuningConfig):
#     pass

# a_args = transformers.HfArgumentParser((peft.PromptTuningConfig)).parse_args_into_dataclasses()
arg_parser = transformers.HfArgumentParser((peft.LoraConfig))
aaa = arg_parser.parse_args_into_dataclasses()

print(aaa)
