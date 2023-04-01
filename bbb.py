
import transformers
import peft


# a_args = transformers.HfArgumentParser((peft.PromptTuningConfig)).parse_args_into_dataclasses()
arg_parser = transformers.HfArgumentParser((peft.LoraConfig))
arg_parser.parse_args_into_dataclasses()

print(b_args)
