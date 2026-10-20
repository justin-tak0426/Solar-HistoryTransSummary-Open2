---
base_model: upstage/Solar-Open2-250B
library_name: peft
pipeline_tag: text-generation
license: other
license_name: upstage-solar-license
license_link: LICENSE.md
language:
- ko
tags:
- lora
- solar_open2
---

# Solar-HistoryTransSummary-Open2-B200

**Built with Solar**

LoRA fine-tuning of `upstage/Solar-Open2-250B` for historical-document translation
and summarization into contemporary Korean.

Base revision: `8d1dda5ad2af58207c65348d825772d2f4d72481`.
This adapter modifies the base model through LoRA training: rank 32, alpha 64,
dropout 0.05, one epoch, 9,316 optimizer steps. The final checkpoint was also
selected as the best checkpoint by full-validation selection loss (0.471465).
The original backbone is required separately to use this adapter.

Distributed under the [Upstage Solar License](LICENSE.md); see [NOTICE](NOTICE)
for attribution and modification details. This derivative model is not an
official Upstage product and is not sponsored, endorsed, or certified by Upstage.
See the [repository README](../README.md) for installation, inference,
training details, validation results, and limitations.
