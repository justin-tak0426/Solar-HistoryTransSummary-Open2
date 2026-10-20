# Solar Open 2 environment

All download, conversion, training, inference, scoring, and W&B commands use:

`/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/solar-open-2/bin/python`

The environment lives beside the executable source, not below the model/data
root. It is a self-contained Python 3.11 prefix with the validated CUDA 12.8
PyTorch stack and all Solar-specific packages stored below that prefix. It does
not import packages from another user's environment, the Qwen environment, or
system site-packages. `solar-open-2.lock.txt` records the exact resolved package
set and is hashed into every model preflight, smoke receipt, and resumable
training state.

Required Solar-specific pins:

- Upstage Transformers commit
  `ef5c37485091057070b36dd0620798f74988eb01`
- `bitsandbytes==0.49.2`
- `accelerate==1.14.0`
- `peft==0.19.1`
- `fla-core==0.5.1`
- `wandb==0.28.0`

The job scripts fail closed unless both `sys.prefix` and `sys.base_prefix`
resolve to this exact `solar-open-2` directory, every core module is imported
from below it, `pip freeze` matches the lock exactly, `pip check` passes, and the
Upstage Transformers VCS commit is unchanged.
