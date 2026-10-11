"""Train the Vampire first-frame + 20-prompt video world model."""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import accelerate
import torch
from safetensors.torch import load_file

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _path in (str(_HERE), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data.profiles import VAMPIRE  # noqa: E402
from data.prompt_utils import resolve_temporal_prompts  # noqa: E402
from data.vampire_dataset import VampireDataset  # noqa: E402
from runner import launch_training_task  # noqa: E402

from diffsynth.diffusion import *  # noqa: E402,F401,F403
from diffsynth.models.world_model_dit import WorldModel  # noqa: E402
from diffsynth.pipelines.world_model import (  # noqa: E402
    ModelConfig,
    WorldModelTrainingPipeline,
    model_fn_world_model,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WAN_MODEL_KWARGS = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}

_RESUME_STEP_PATTERN = re.compile(r"(?:step-|state-)(\d+)(?:\.safetensors)?$")


def _resume_step_from_path(path: str | None) -> int:
    if not path:
        return 0
    match = _RESUME_STEP_PATTERN.search(Path(path).name)
    return int(match.group(1)) if match else 0


def _reserve_cuda_memory(gb: float) -> None:
    if gb <= 0 or not torch.cuda.is_available():
        return
    device = torch.device("cuda", torch.cuda.current_device())
    target_bytes = int(gb * 1024**3)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes <= target_bytes:
        raise RuntimeError(
            f"cannot reserve {gb:.1f} GiB on {device}: only "
            f"{free_bytes / 1024**3:.1f} GiB free of {total_bytes / 1024**3:.1f} GiB"
        )
    reservation = torch.empty(target_bytes // 2, dtype=torch.float16, device=device)
    reservation.zero_()
    del reservation
    torch.cuda.synchronize(device)


def flow_match_video_loss(pipe: WorldModelTrainingPipeline, **inputs):
    """Flow-matching MSE over only the 80 predicted frames."""
    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)
    max_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_boundary, max_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(
        dtype=pipe.torch_dtype, device=pipe.device,
    )
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(
        inputs["input_latents"], noise, timestep,
    )
    target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, :1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
    if "first_frame_latents" in inputs:
        prediction = prediction[:, :, 1:]
        target = target[:, :, 1:]
    loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
    return loss * pipe.scheduler.training_weight(timestep)


class FullModelLogger(ModelLogger):  # noqa: F405
    def __init__(self, output_path, initial_step=0):
        super().__init__(output_path, remove_prefix_in_ckpt="pipe.dit.")
        self.num_steps = int(initial_step)


class WorldModelTrainingModule(DiffusionTrainingModule):  # noqa: F405
    def __init__(
        self,
        profile,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        trainable_filter="",
        trainable_filter_exclude="",
        resume_from_ckpt=None,
        prompt_dropout_prob=0.0,
    ):
        super().__init__()
        self.profile = profile
        if not use_gradient_checkpointing:
            warnings.warn("gradient checkpointing is disabled; watch for OOM")
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/",
            )
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        self.pipe = WorldModelTrainingPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        custom_dit = WorldModel(**WAN_MODEL_KWARGS).to(torch.bfloat16)
        custom_dit = self._transfer_weights(custom_dit, self.pipe.dit)
        if resume_from_ckpt:
            checkpoint = load_file(resume_from_ckpt)
            missing, unexpected = custom_dit.load_state_dict(checkpoint, strict=False)
            if unexpected:
                raise ValueError(
                    "resume checkpoint is not a video-only world-model checkpoint; "
                    f"unexpected keys include {unexpected[:5]}"
                )
            if missing:
                raise ValueError(
                    f"resume checkpoint is incomplete; missing keys include {missing[:5]}"
                )
            print(f"[Resume] loaded {len(checkpoint)} tensors from {resume_from_ckpt}")
            del checkpoint
        self.pipe.dit = custom_dit.to(self.pipe.device)
        del custom_dit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.pipe.model_fn = model_fn_world_model
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model,
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )
        if trainable_models:
            for name in trainable_models.split(","):
                module = self.pipe.get_module(self.pipe, name)
                if module is not None:
                    module.train()
                    module.requires_grad_(True)
        self._apply_trainable_filter(trainable_filter, trainable_filter_exclude)
        self.prompt_dropout_prob = float(prompt_dropout_prob)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.fp8_models = fp8_models
        self.task = task
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    @staticmethod
    def _transfer_weights(custom_model, pretrained_dit):
        pretrained = pretrained_dit.state_dict()
        custom = custom_model.state_dict()
        compatible = {
            key: value for key, value in pretrained.items()
            if key in custom and value.shape == custom[key].shape
        }
        required = (
            "patch_embedding.weight",
            "text_embedding.0.weight",
            "time_embedding.0.weight",
            "blocks.0.self_attn.q.weight",
            "blocks.0.cross_attn.q.weight",
            "blocks.0.ffn.0.weight",
            "head.head.weight",
        )
        absent = [key for key in required if key not in compatible]
        if absent:
            raise RuntimeError(
                "Wan cold-start did not provide required weights: " + ", ".join(absent)
            )
        missing, unexpected = custom_model.load_state_dict(compatible, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected cold-start keys: {unexpected[:5]}")
        print(
            f"[Cold-start] transferred {len(compatible)}/{len(pretrained)} Wan tensors; "
            f"missing={len(missing)}"
        )
        return custom_model

    def _apply_trainable_filter(self, keep: str, drop: str):
        keep_patterns = [value.strip() for value in keep.split(",") if value.strip()]
        drop_patterns = [value.strip() for value in drop.split(",") if value.strip()]
        if not keep_patterns and not drop_patterns:
            return
        for name, parameter in self.pipe.dit.named_parameters():
            if keep_patterns and not any(pattern in name for pattern in keep_patterns):
                parameter.requires_grad = False
            if any(pattern in name for pattern in drop_patterns):
                parameter.requires_grad = False
        scalar_count = sum(
            parameter.numel() for parameter in self.pipe.dit.parameters()
            if parameter.requires_grad
        )
        print(f"[Trainable Filter] trainable scalars={scalar_count:,}")

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for name in extra_inputs:
            if name == "input_image":
                inputs_shared[name] = data["video"][0]
            elif name == "end_image":
                inputs_shared[name] = data["video"][-1]
            elif name in ("reference_image", "vace_reference_image"):
                inputs_shared[name] = data[name][0]
            else:
                inputs_shared[name] = data[name]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        prompts = resolve_temporal_prompts(data, self.profile, expected_count=20)
        inputs_positive = {"prompt": prompts}
        inputs_negative = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "prompt_num_frames": int(data.get("source_num_frames", 80)),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_positive, inputs_negative

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(
            inputs, self.pipe.device, self.pipe.torch_dtype,
        )
        if (
            self.prompt_dropout_prob > 0
            and torch.rand(1).item() < self.prompt_dropout_prob
        ):
            inputs_shared, inputs_positive, inputs_negative = inputs
            inputs_positive["prompt"] = [""] * 20
            inputs = inputs_shared, inputs_positive, inputs_negative
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_positive, _ = inputs
        return flow_match_video_loss(self.pipe, **inputs_shared, **inputs_positive)


def build_parser():
    parser = argparse.ArgumentParser(description="Vampire video world-model training")
    parser = add_general_config(parser)  # noqa: F405
    parser = add_video_size_config(parser)  # noqa: F405
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--initialize_model_on_cpu", action="store_true")
    parser.add_argument("--trainable_filter", type=str, default="")
    parser.add_argument("--trainable_filter_exclude", type=str, default="")
    parser.add_argument("--resume_from_ckpt", type=str, default=None)
    parser.add_argument("--prompt_dropout_prob", type=float, default=0.1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--skip_final_save", action="store_true")
    parser.add_argument("--reserve_cuda_memory_gb", type=float, default=0.0)
    parser.add_argument("--save_full_state", action="store_true")
    parser.add_argument("--resume_state", type=str, default=None)
    return parser


class _MultiGameDataset(torch.utils.data.ConcatDataset):
    """ConcatDataset that still exposes the `load_from_cache` flag runner.py checks."""

    load_from_cache = False


def build_dataset(args):
    base_paths = [value.strip() for value in args.dataset_base_path.split(";") if value.strip()]
    metadata_paths = [value.strip() for value in args.dataset_metadata_path.split(";") if value.strip()]
    if len(base_paths) <= 1 and len(metadata_paths) <= 1:
        return VampireDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            height=args.height,
            width=args.width,
            max_pixels=args.max_pixels,
            num_frames=args.num_frames,
            expected_prompts=20,
        )
    if len(base_paths) != len(metadata_paths):
        raise ValueError(
            "--dataset_base_path and --dataset_metadata_path must list the same "
            f"number of ';'-separated entries; got {len(base_paths)} base paths "
            f"and {len(metadata_paths)} metadata paths"
        )
    datasets = [
        VampireDataset(
            base_path=base_path,
            metadata_path=metadata_path,
            repeat=args.dataset_repeat,
            height=args.height,
            width=args.width,
            max_pixels=args.max_pixels,
            num_frames=args.num_frames,
            expected_prompts=20,
        )
        for base_path, metadata_path in zip(base_paths, metadata_paths)
    ]
    for base_path, dataset in zip(base_paths, datasets):
        print(f"[MultiGameDataset] {base_path}: {len(dataset)} rows")
    return _MultiGameDataset(datasets)


def main():
    args = build_parser().parse_args()
    if args.resume_state and args.resume_from_ckpt:
        raise SystemExit("--resume_state and --resume_from_ckpt are mutually exclusive")
    print(
        "[WorldModel] input=first_frame+20_prompts output=80_frames@16fps "
        "duration=5s video_latents=21"
    )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=args.find_unused_parameters,
        )],
    )
    _reserve_cuda_memory(args.reserve_cuda_memory_gb)
    dataset = build_dataset(args)
    model = WorldModelTrainingModule(
        profile=VAMPIRE,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        trainable_filter=args.trainable_filter,
        trainable_filter_exclude=args.trainable_filter_exclude,
        resume_from_ckpt=args.resume_from_ckpt,
        prompt_dropout_prob=args.prompt_dropout_prob,
    )
    initial_step = _resume_step_from_path(args.resume_state or args.resume_from_ckpt)
    model_logger = FullModelLogger(args.output_path, initial_step=initial_step)
    if accelerator.is_main_process and initial_step:
        print(f"[Resume] continuing checkpoint numbering from step {initial_step}")
    launcher_map = {
        "sft:data_process": launch_data_process_task,  # noqa: F405
        "direct_distill:data_process": launch_data_process_task,  # noqa: F405
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
