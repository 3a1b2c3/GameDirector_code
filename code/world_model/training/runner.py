"""Training launcher with lightweight checkpoints and optional full resume."""
from __future__ import annotations

import os
from typing import Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from diffsynth.diffusion.training_module import DiffusionTrainingModule


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: Optional[int] = None,
    num_epochs: int = 1,
    args=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    max_train_steps = None if args is None else getattr(args, "max_train_steps", None)
    skip_final_save = bool(args is not None and getattr(args, "skip_final_save", False))
    save_full_state = bool(args is not None and getattr(args, "save_full_state", False))
    resume_state_path = None if args is None else args.resume_state

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda batch: batch[0], num_workers=num_workers,
    )
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler,
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)

    if resume_state_path:
        if not os.path.isdir(resume_state_path):
            raise SystemExit(f"--resume_state is not a directory: {resume_state_path}")
        accelerator.load_state(resume_state_path)
        if accelerator.is_main_process:
            print(f"[Resume] restored complete training state from {resume_state_path}")

    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                loss = model({}, inputs=data) if dataset.load_from_cache else model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)

                if max_train_steps is not None and model_logger.num_steps >= max_train_steps:
                    if accelerator.is_main_process:
                        print(f"[Max-Steps] reached {max_train_steps}", flush=True)
                    if not skip_final_save:
                        model_logger.on_training_end(accelerator, model, save_steps)
                    accelerator.end_training()
                    return

                if (
                    save_full_state
                    and save_steps is not None
                    and model_logger.num_steps % save_steps == 0
                ):
                    output_dir = os.path.join(
                        model_logger.output_path, f"state-{model_logger.num_steps}",
                    )
                    if accelerator.is_main_process:
                        os.makedirs(output_dir, exist_ok=True)
                    accelerator.wait_for_everyone()
                    accelerator.save_state(output_dir)
                    if accelerator.is_main_process:
                        print(f"[Save-State] wrote {output_dir}", flush=True)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    if not skip_final_save:
        model_logger.on_training_end(accelerator, model, save_steps)
    accelerator.end_training()


__all__ = ["launch_training_task"]
