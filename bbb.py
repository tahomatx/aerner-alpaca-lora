
import sys
import os
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

# # @dataclass
# # class PeftArguments(peft.LoraConfig, peft.PromptTuningConfig):
# #     pass


# # a_args = transformers.HfArgumentParser((peft.PromptTuningConfig)).parse_args_into_dataclasses()
# arg_parser = transformers.HfArgumentParser((transformers.TrainingArguments))
# aaa = arg_parser.parse_args_into_dataclasses()

# print(aaa)


parser = transformers.HfArgumentParser((peft.LoraConfig))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.abspath(sys.argv[1]))
else:
    peft_args = parser.parse_args_into_dataclasses()
