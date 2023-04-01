
import transformers
import peft
import datasets
from dataclasses import dataclass, field


@dataclass
class PeftArguments:
    peft_type: str = field()
    r: int = field()
    lora_alpha: int = field(),
    lora_dropout: float = field(),
    # Used for prompt tuning, prefix tuning and p-tuning
    num_virtual_tokens: int = field()
    mapping_hidden_dim: int = field()

@dataclass
class PeftArguments(peft.LoraConfig, peft.PromptTuningConfig):
    pass

# a_args = transformers.HfArgumentParser((peft.PromptTuningConfig)).parse_args_into_dataclasses()
arg_parser = transformers.HfArgumentParser((PeftArguments))
aaa = arg_parser.parse_args_into_dataclasses()

print(aaa)
