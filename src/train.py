# I'm too tired to do this properly, just gonna copy paste from the Pyg training code repo
import pathlib
import typing as t

from dataclasses import dataclass, field

import torch
import transformers

from datasets import MmappedArrowDataset, DataCollatorForMmapedDataset
from mezo import MeZOTrainer

@dataclass
class ModelArguments:
    model_name_or_path: t.Optional[str] = field(
        default="EleutherAI/pythia-70m-deduped")

@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Path to the training set."})
    eval_file: str = field(metadata={"help": "Path to the evaluation set."})


@dataclass
class OtherArguments:
    model_load_delay_per_rank: t.Optional[int] = field(metadata={
        "help": "Delay loading the model by (this many seconds) * (local_rank)."},
        default=None)
    add_special_tokens: t.Optional[str] = field(
        metadata={"help": "Extra special tokens to add to the tokenizer before training. Comma-separated."},
        default=None)
    uft: bool = field(
        metadata={"help": "Use unsupervised fine-tuning instead of supervised fine-tuning."},
        default=False)
    
@dataclass
class MezoArguments:
    use_mezo: bool = field(
        metadata={"help": "Use the MeZO optimizer."},
        default=True
    )
    zo_eps: float = field(
        metadata={"help": "EPS for MeZO."},
        default=1e-3
    )
    linear_probing: bool = field(
        metadata={"help": "Use linear probing."},
        default=False
    )
    lp_early_stopping: bool = field(
        metadata={"help": "Stop linear probing early."},
        default=False
    )

def main() -> None:
    parser = transformers.HfArgumentParser((
        ModelArguments,
        DataArguments,
        OtherArguments,
        MezoArguments,
        transformers.TrainingArguments,
    ))
    model_args, data_args, other_args, mezo_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
    )

    if other_args.model_load_delay_per_rank is not None:
        # See comment in PygmalionAI/training-code
        import time
        time.sleep(other_args.model_load_delay_per_rank * training_args.local_rank)

    # Model loading.
    model_load_dtype = None
    if training_args.bf16:
        model_load_dtype = torch.bfloat16
    elif training_args.fp16:
        model_load_dtype = torch.float16

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path
    )

    model = transformers.AutoModelForCausalLM.from_config(config).cuda()

    if other_args.add_special_tokens is not None:
        # MAINTENANCE(11b): Big fat warning: the snippet below is copy-pasted
        # into ``./preparation/tokenize_data_{sft,uft}.py``. Make sure to always keep both
        # implementations in sync.
        special_token_contents = other_args.add_special_tokens.split(",")
        special_tokens = [
            transformers.AddedToken(
                # Heads up: this is very poorly documented in HuggingFace and
                # some old forum discussions mention that it's apparently
                # exclusive to the Rust-based tokenizers? If anything seems
                # funky about the special token behavior, this is a good place
                # to look.
                content, lstrip=True, rstrip=True)
            for content in special_token_contents
        ]

        _add_special_tokens_to_tokenizer_and_resize_model_embeddings(
            {"additional_special_tokens": special_tokens},
            tokenizer,
            model,
        )

    # Silence this annoying warning.
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    # Dataset setup.
    train_dataset = MmappedArrowDataset(data_args.train_file, sft=not other_args.uft)
    eval_dataset = MmappedArrowDataset(data_args.eval_file, sft=not other_args.uft)
    data_collator = DataCollatorForMmapedDataset(tokenizer=tokenizer, sft=not other_args.uft)

    trainer = MeZOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=training_args,
        use_mezo=mezo_args.use_mezo,
        zo_eps=mezo_args.zo_eps,
        linear_probing=mezo_args.linear_probing,
        lp_early_stopping=mezo_args.lp_early_stopping,
        callbacks=None,
    )

    try:
        # Resume from checkpoint if we have any checkpoints automatically saved
        # by the HF Trainer within the output directory.
        resume_from_checkpoint = len(
            list(pathlib.Path(
                training_args.output_dir).glob("checkpoint-*"))) > 0

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    except KeyboardInterrupt as ex:
        # TODO(11b): Test whether this does what I expect. Idea is to have the
        # trainer save the current state when I interrupt the run so I don't
        # need to keep waiting for a checkpoint step.
        # trainer.save_model()
        # trainer.save_state()
        raise ex

    trainer.save_state()
    trainer.save_model()

def _add_special_tokens_to_tokenizer_and_resize_model_embeddings(
    special_tokens: t.Dict[str, t.Union[str, transformers.AddedToken]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
):
    tokenizer.add_special_tokens(special_tokens)

    # Size is rounded up to the nearest number divisible by 64 for performance
    # reasons.
    new_size = _nearest_divisible(num=len(tokenizer), divisor=64)
    old_size = model.config.vocab_size

    if new_size == old_size:
        # No resizing needs to be done, let's bail!
        return

    # Need to resize the token embeddings. We initialize the new positions with
    # the mean of the existing ones to cut down on required training time.
    model.resize_token_embeddings(new_size)
    new_positions_count = new_size - old_size

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    # This is just to keep the LSP happy.
    assert isinstance(input_embeddings, torch.Tensor)
    assert isinstance(output_embeddings, torch.Tensor)

    input_embeddings_avg = input_embeddings[:-new_positions_count].mean(dim=0,
                                                             keepdim=True)
    output_embeddings_avg = output_embeddings[:-new_positions_count].mean(dim=0,
                                                               keepdim=True)

    input_embeddings[-new_positions_count:] = input_embeddings_avg
    output_embeddings[-new_positions_count:] = output_embeddings_avg


def _nearest_divisible(num: int, divisor: int) -> int:
    '''Returns the nearest number to `num` that is divisible by `divisor`.'''
    return (num + divisor - 1) // divisor * divisor

if __name__ == "__main__":
    main()
