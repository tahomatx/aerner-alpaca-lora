
import transformers
import peft

all_args, prompt_tuning_config, prefix_tuning_config, prompt_encoding_config, lora_config = transformers.HfArgumentParser(
    (peft.PromptTuningConfig, peft.PrefixTuningConfig, peft.PromptEncoderConfig, peft.LoraConfig)).parse_args_into_dataclasses()

print(all_args, prompt_tuning_config, prefix_tuning_config,
      prompt_encoding_config, lora_config)
