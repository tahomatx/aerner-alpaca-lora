
import transformers
import peft
import datasets
from dataclasses import dataclass, field



@dataclass
class PeftArguments(peft.LoraConfig, peft.PromptTuningConfig):
    pass

# a_args = transformers.HfArgumentParser((peft.PromptTuningConfig)).parse_args_into_dataclasses()
arg_parser = transformers.HfArgumentParser((PeftArguments))
aaa = arg_parser.parse_args_into_dataclasses()

print(aaa)
