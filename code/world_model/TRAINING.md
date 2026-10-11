# Training contract

- Each training item is 80 frames, 16 FPS, 20 prompts; no state file is read.
- The first frame is duplicated to form 81 frames; the VAE encodes this to 21
  latents, with latent 0 fixed as the clean conditioning frame.
- The flow-matching MSE loss only supervises latents 1..20.
- The 20 UMT5 prompt embeddings and the 20 predicted latents are aligned
  one-to-one via local cross-attention (Sec. 3.4, Eq. 6).
- `step-N.safetensors` saves the full DiT; an optional full Accelerate resume
  state is also saved.
- Cold start loads all shape-compatible Wan2.2 video weights from the directory
  selected by `GAMEDIRECTOR_BASE_MODEL` or `WORLD_MODEL_BASE_MODEL`.
