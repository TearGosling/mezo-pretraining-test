# Attempt 2 of implementing MeZO,
# this time with a proper class rather than fully modifying a training loop
import typing as t

import numpy as np
import torch

from torch import nn

from train import MeZOTrainer

IGNORE_DECAY_PARAMS = ["bias", "layer_norm", "layernorm"]

class MeZOOptimizer():
    def __init__(self, trainer: MeZOTrainer) -> None:
        '''
        MeZO optimizer.
        TODO: Actually put something here later.
        '''
        self.trainer = trainer
        self.learning_rate = trainer.learning_rate
        self.eps = trainer.eps
        self.weight_decay = trainer.weight_decay
        self.n_gpus = torch.cuda.device_count()

        # Sanity checks
        if self.learning_rate < 0.0:
            raise ValueError(f"LR {self.learning_rate} is less than or equal to zero!")
        if self.eps < 0.0:
            raise ValueError(f"EPS {self.eps} is less than or equal to zero!")

        # Values
        self.zo_random_seed = None
        self.projected_grad = None

    def step(self, inputs) -> torch.tensor:
        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self._perturb_parameters(scaling_factor=1)
        loss1 = self._get_loss(inputs)

        # Second function evaluation
        self._perturb_parameters(scaling_factor=-2)
        loss2 = self._get_loss(inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.eps)).item()

        self._perturb_parameters(scaling_factor=1)

        return loss1
    
    def update(self) -> None:
        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                # Resample z
                z = self._sample_z(p)
                if _list_not_in_name(search_terms=IGNORE_DECAY_PARAMS, name=name):
                    p.data = p.data - self.learning_rate * (self.projected_grad * z + self.weight_decay * p.data)
                else:
                    p.data = p.data - self.learning_rate * (self.projected_grad * z)

    def _perturb_parameters(
            self,
            random_seed: t.Optional[int] = None,
            scaling_factor: float = 1,
    ) -> None:
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is None else self.zo_random_seed)

        for p in self.model.parameters():
            if p.requires_grad:
                z = self._sample_z(p)
                p.data = p.data + scaling_factor * z * self.eps

    def _get_loss(self, inputs) -> torch.tensor:
        with torch.inference_mode():
            loss = self.trainer.model(**inputs).loss
            if self.n_gpus > 1:
                loss = loss.mean()

        return loss.detach()
    
    def _sample_z(self, param: nn.Parameter) -> torch.tensor:
        return torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

def _list_not_in_name(search_terms: list, name: str) -> bool:
    for t in search_terms:
        if t in name:
            return False
    return True
