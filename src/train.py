import argparse
import functools
import json
import logging
import os
import time

import accelerate
import pyarrow
import torch
import transformers

from tqdm import tqdm

from datasets import MmappedArrowDataset, uft_collate_fn
from mezo_op import MeZOOptimizer

LOG = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    level=logging.DEBUG,
)

# Convert any number to GiB
GiB = lambda x: x >> 30
# Get the GPU ID
GPU_ID = torch.cuda.device().split(":")[-1]

STR_TO_TORCH_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}

class MeZOTrainer:
    def __init__(
        self,
        args: argparse.Namespace
    ) -> None:
        # Assignment
        self.args = args
        self.num_epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.global_step = 0
        self.local_step = 0

        # Load the model and tokenizer up
        LOG.info("Loading tokenizer...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
        LOG.info("Done! Loading model...")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name,
            low_cpu_mem_usage=True,
            torch_dtype=STR_TO_TORCH_DTYPE[args.dtype]
        ).cuda()
        self.model.eval()
        LOG.info(f"Done! Model and tokenizer initialized on GPU {GPU_ID} with {GiB(torch.cuda.memory_allocated):.3f} GiB of allocated memory.")

        # Load optimizer
        self.optimizer = MeZOOptimizer(
            # Surely this won't cause any problems
            # (clueless)
            trainer=self
        )

        train_dataset = MmappedArrowDataset(args.train_dataset_path, sft=False)
        eval_dataset = MmappedArrowDataset(args.eval_dataset_path, sft=False)
        collate_fn = functools.partial(uft_collate_fn, tokenizer=self.tokenizer)

        # Load dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        self._init_accelerator()

        if self.accelerator.is_main_process:
            self.progress_bar = tqdm(
                total=len(self.train_dataloader)*len(self.eval_dataloader),
                desc="Training/eval steps",
                leave=False
            )

    def train(self):
        '''MeZO training.'''
        if self.accelerator.is_main_process:
            LOG.info("Beginning training loop!")
        
        hps = {
            "base_model": self.model_name,
            "lr": self.learning_rate,
        }
        self.accelerator.init_trackers(self.args.project_name, config=hps)

        for epoch in range(self.num_epochs):
            LOG.info(f"Beginning epoch {epoch} of training.")
            for i, batch in enumerate(self.train_dataloader):
                step_start = time.perf_counter()
                metrics = self.step(batch, loss_type="train")
                step_end = time.perf_counter()
                self.local_step += 1

                if self.accelerator.is_main_process:
                    rank_samples_per_second = self.args.batch_size / (step_end - step_start)
                    world_samples_per_second = rank_samples_per_second * self.accelerator.num_processes

                    metrics.update({
                        "perf/rank_samples_per_second": rank_samples_per_second,
                        "perf/world_samples_per_second": world_samples_per_second,
                        "train/epoch": epoch,
                        "train/samples_seen": self.local_step * self.args.batch_size,
                    })

                    self.progress_bar.update(1)
                    self.progress_bar.set_postfix(**metrics)

                    self.accelerator.log(metrics, step=self.local_step)

                if self.local_step % self.args.save_steps == 0:
                    self.save_model()
                    # eval
                    eval_losses = []
                    if self.accelerator.is_main_process:
                        progress_bar = tqdm(
                            total=len(self.eval_dataloader),
                            desc="Eval steps",
                            leave=False
                        )

                    for e_batch in self.eval_dataloader:
                        loss = self.step(e_batch, loss_type="eval")
                        loss_type = "eval/loss"

                        eval_losses.append(loss[loss_type])
                        if self.accelerator.is_main_process:
                            progress_bar.update(1)

                    eval_losses = torch.cat(eval_losses)
                    eval_losses = eval_losses[:len(self.eval_dataloader)]
                    eval_loss = torch.mean(eval_losses)
                    if self.accelerator.is_main_process:
                        self.accelerator.log({loss_type: eval_loss}, step=self.local_step)

        LOG.info("Training complete! Saving model...")
        self.save_model()
        self.accelerator.end_training()

    def save_model(self):
        save_path = os.path.join(self.args.save_path, self.args.wandb_project_name)
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)

            state_file_path = os.path.join(save_path, "trainer_state.json")
            with open(state_file_path, "w") as state_file:
                state_file.write(json.dumps({
                    "step": self.local_step()
                }))

            self.accelerator.wait_for_everyone()

            self.accelerate.save_state(save_path)

    def load_from_checkpoint(self):
        raise NotImplementedError

    def _init_accelerator(self) -> None:
        '''Initializes the accelerator and prepares the model and dataloaders.'''
        self.accelerator = accelerate.Accelerator()
        accelerate.utils.set_seed(42)
        self.model, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.model, self.train_dataloader, self.eval_dataloader
        )

    def step(self, batch: dict, loss_type: str) -> dict:
        '''Singular training step which can be shared with train or eval.'''
        with torch.no_grad():
            loss = self.optimizer.step(batch)
            self.optimizer.update()

        return {
            f"{loss_type}/loss": loss
        }

def _parse_args_from_argv() -> argparse.Namespace:
    '''Parses arguments'''
    parser = argparse.ArgumentParser(prog="MeZO Trainer", description="Trains/fine-tunes a model using the MeZO algorithm")

    ### MODEL ARGS ###
    parser.add_argument(
        "--model-name",
        required=True,
        help="The HuggingFace model name or path to a local checkpoint"
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        help="The dtype to train the model in"
    )

    ### TRAINING ARGS ###
    parser.add_argument(
        "--wandb-project-name",
        default="new-mezo-trainer",
        help="The name of the wandb project."
    )
    parser.add_argument(
        "--save-steps",
        required=True,
        help="When to save the model"
    )

    ### HYPERPARAMETERS ###
    parser.add_argument(
        "--batch-size",
        required=True,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        required=True,
        help="Learning rate"
    )
    parser.add_argument(
        "--eps",
        default=1e-3,
        help="EPS"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.,
        help="Weight decay"
    )

    ### PATHS ###
    parser.add_argument(
        "--train-dataset",
        required=True,
        help="Path to the pyarrow file for the training dataset"
    )
    parser.add_argument(
        "--eval-dataset",
        required=True,
        help="Path to the pyarrow file for the eval dataset"
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="Save the model in this directory"
    )

    return parser.parse_args()

def main() -> None:
    args = _parse_args_from_argv()
    trainer = MeZOTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
