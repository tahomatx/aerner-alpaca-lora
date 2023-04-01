
import transformers
import peft


# a_args = transformers.HfArgumentParser((peft.PromptTuningConfig)).parse_args_into_dataclasses()
b_args = transformers.HfArgumentParser((peft.LoraConfig)).parse_args_into_dataclasses()

print(b_args)
