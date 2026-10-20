# Solar Open 2 routed-expert NF4

This directory implements a fail-closed, experts-only NF4 sidecar for the
official `upstage/Solar-Open2-250B` revision
`8d1dda5ad2af58207c65348d825772d2f4d72481`.

It never downloads model files or submits jobs. The only production artifact
root accepted by the converter is:

`/data1/jhtak/Solar-HistoryTransSummary-Open2`

The source snapshot default is:

`/data1/jhtak/Solar-HistoryTransSummary-Open2/hf_cache/models--upstage--Solar-Open2-250B/snapshots/8d1dda5ad2af58207c65348d825772d2f4d72481`

## Default layout

`convert_layer_flat.py` is the production default. The official source has
46,080 per-expert routed matrices:

- `gate_proj` and `up_proj`: `[1280, 4096]` per expert
- `down_proj`: `[4096, 1280]` per expert

Conversion preserves expert order and concatenates 64 experts at a time. Each
layer therefore stores five `gate_up_proj` chunks and five `down_proj` chunks.
The largest chunk contains 671,088,640 elements, below the bitsandbytes CUDA
`INT_MAX` boundary; quantizing a whole layer's 3,355,443,200-element
`gate_up_proj` matrix is deliberately rejected.

No transpose or expert reorder is allowed. At runtime the ten chunks are
dequantized sequentially into the official 3-D layer tensors before native
expert dispatch. The temporary BF16 routed weights are 10,066,329,600 bytes
(9.375 GiB) per active layer. Decoder gradient checkpointing is mandatory, and
the implementation does not cache dequantized layers.

The sidecar is not a complete Hugging Face checkpoint. The sealed manifest
sets `complete_model_checkpoint=false`.

Read-only plan:

```bash
/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/solar-open-2/bin/python \
  -m scripts.quantization.convert_layer_flat
```

Conversion, only after an allocated SM120 GPU is available and the source
snapshot is complete:

```bash
/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/solar-open-2/bin/python \
  -m scripts.quantization.convert_layer_flat --execute --device cuda:0
```

The command requires:

- Upstage Transformers `v5.14.1-solar-open2`
- `bitsandbytes==0.49.2`
- `accelerate==1.14.0`
- CUDA PyTorch and an SM120+ GPU

It verifies the pinned source index (SHA-256
`2ba6aa72c31d2d758f747a546df3d671b5e1d5f1fa82985e789117b9cb148430`),
all 94 source shard headers and hashes, all 47,019 source tensor keys,
`metadata.total_size=500575651328`, the exact official meta-model schema, the
250,287,794,944 parameter contract, and NF4 error bounds before sealing. The
sealed layer-flat sidecar has 2,880 serialized tensors.

The official config has `tie_word_embeddings=false`. The loader therefore
streams `model.embed_tokens.weight` and `lm_head.weight` independently and
rejects aliasing between them.

## Full loader

`full_model_loader.py` creates the official model on meta, replaces native
experts before any BF16 expert is materialized, streams every non-expert tensor
to contiguous 16-layer blocks on three GPUs, and installs differentiable input
transfer hooks. Attention and shared-expert LoRA targets remain BF16
`torch.nn.Linear`; only routed experts are NF4.

The loader always returns `production_authorized=false`. A real maximum-length
one-update forward/backward, adapter save/reload, and peak-memory smoke remains
the next gate.

## Fallback

`convert.py` and `expert_module.py` retain the per-expert Linear4bit fallback.
It produces 46,080 independent quantized linear modules and is not the default
because module/kernel launch overhead is expected to be substantially higher.

## Tests

```bash
/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/solar-open-2/bin/python \
  -m unittest scripts.quantization.test_quantization -v
```

CPU tests prove the exact split mapping, dense forward and input-gradient
fidelity, bounds failures, deterministic plans, and header accounting. The
bitsandbytes test is skipped unless `bitsandbytes==0.49.2` and an allocated
SM120 GPU are both present; that skip prevents a CPU-only pass from authorizing
production.
