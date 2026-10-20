# Solar-HistoryTransSummary-Open2

**Built with Solar**

Solar Open 2 기반 근대 사료 번역·요약 LoRA 튜닝 프로젝트다. 모델 이름은
Upstage Solar License 4(e)에 맞춰 `Solar`로 시작한다. 이 저장소에는 모델
weight를 넣지 않고, 전용 `/data1/jhtak` runtime과 Slurm dependency chain으로
download, NF4 변환, smoke, 학습, 평가를 실행한다.

## Current state

- 공식 기반 모델: `upstage/Solar-Open2-250B`
- 고정 revision: `8d1dda5ad2af58207c65348d825772d2f4d72481`
- 학습 runtime: 고정 BF16 snapshot의 routed experts만 custom NF4 sidecar로 변환
- NF4 training artifact contract:
  `solar-open2-layer-flat-routed-experts-nf4` schema v1, `SEALED`
- recipe/state/runtime contract: recipe v3 early-stop, state v4, runtime projection
  schema v5
- 구현 상태: 최대 2-epoch/9,600-token 조기 종료 설계 구현 및 smoke 검증 중
- production gate: v7 3-GPU max-length/조기 종료 smoke 통과와 두 번째 명시적
  승인 전

원본 BF16 weight는 약 500.6 GB이므로 3 x 96 GB GPU에서 DDP로 복제할 수 없다.
이 프로젝트는 3 GPU single-process model parallel을 사용한다. 공식 BF16
snapshot에서 routed expert projection만 bitsandbytes NF4로 변환하고,
attention/shared expert/lm head 등 나머지 weight는 같은 고정 BF16 snapshot에서
직접 적재한다. Nota NVFP4는 inference-only 후보로만 기록하며 학습 runtime으로
허용하지 않는다. `preflight_model.py`와 새 9,600-token GPU smoke가 모두 통과한
영수증이 없으면 production script가 종료한다. 현재 production 제출은 승인되지
않았다.

## Fixed recipe

| Item | Value |
| --- | ---: |
| Maximum augmented-deck epochs | 2 |
| Optimizer updates | earliest stop 13,973 / maximum 18,631 |
| Micro batch | length bucket별 target 8 / 4 / 2 / 1 |
| Gradient accumulation | optimizer group당 16 occurrence가 되도록 동적 |
| Effective batch | 16 |
| Max sequence length | 9,600 |
| Learning rate | peak 1e-4, cosine floor 1e-5 |
| Scheduler | 3% warmup (558 updates), then cosine with minimum ratio 0.1 |
| LoRA rank / alpha | 32 / 64 |
| Hard-term target reading loss | 2.5x |
| Expert-corrected oversampling | epoch마다 8x |
| NEFTune alpha | 5 |
| Training task mixture | exact deck 자연 비율 68.5846 / 31.4154 |
| Validation selection weight | 0.65 / 0.35 |
| Subset early stopping | minimum step 9,316, patience 3, confirmation 13,973 |
| Checkpoint selection | completed full-val loss argmin, 동률이면 이른 step |

한 epoch은 translation 102,224건과 summarization 46,824건을 합친 149,048
occurrence의 deterministic global augmented deck이다. 최대 horizon은 두 epoch의
298,096 occurrence, 즉 effective batch 16 기준 18,631 optimizer step이다. 승인된
조기 종료가 발생해도 종료 step까지의 deck prefix를 빠짐없이 처리한다. 길이
bucket은 연속된 16-occurrence optimizer group 안에서만 구성하고 다음 group의
row를 당겨오지 않는다. Target microbatch는 1,024 token 이하 8, 2,048 이하 4,
4,096 이하 2, 그보다 길면 1이며 실제 허용치는 smoke 결과로 확정한다.

Learning rate는 peak `1e-4`까지 558 update 동안 warmup한 뒤 전체
18,631-step horizon에서 cosine decay하며 `1e-5` 아래로 내려가지 않는다.
Scheduler는 별도 mutable state를 저장하지 않고 durable `completed_steps`에서
다음 update 위치를 계산하므로, resume 전후 동일한 optimizer step은 동일한
learning rate를 사용한다.

Validation은 train update에 섞지 않는다. 1,000-step마다 고정된
task-stratified subset을 모니터링하고 step 9,316, 13,973, 18,631에서 전체
validation 후보를 측정한다. 최소 1 epoch인 step 9,316 전에는 종료하지 않는다.
step 9,000 subset을 기준으로 절대 `0.001` 또는 상대 `0.25%` 중 큰 폭의 개선이
3회 연속 없더라도 step 13,973 full validation 확인 전에는 종료하지 않는다.
full selection loss가 절대 `0.0005` 또는 상대 `0.1%` 중 큰 폭으로 개선되거나
translation/summary 중 하나가 같은 기준으로 실질 개선되면 계속한다. 그렇지
않으면 최초로 step 13,973에서 종료할 수 있고, 늦어도 step 18,631에서 끝난다.

완료된 full-validation 후보 중 `0.65 * translation_loss + 0.35 *
summarization_loss`가 가장 낮은 `best_adapter`를 선택하며 정확히 같으면 더 이른
step을 선택한다. `final_adapter`는 실제 종료 step을 증명하며, step 18,631에
도달한 경우에만 정확한 2-epoch 완주 증거다. Test와 human gold는 모니터링과
checkpoint 선택에서 완전히 봉인하며 최종 평가에서 별도로 리포팅한다.

상세 산술, resume, runtime, artifact 승인 계약은
[승인 학습 설계](docs/APPROVED_TRAINING_DESIGN_20260727.md)를 따른다.

학습 loss에는 reference prompt token을 전부 mask하고 assistant의 번역·요약 본문과
공식 generation EOS인 `<|im:end|>` 하나만 넣는다. 공식 template이 본문 뒤에
다른 token을 추가하거나 reasoning-disabled assistant preamble이 달라지면 학습
전에 중단한다. 추론은
reasoning을 끄고 EOS/pad protocol token을 제외한 raw continuation에서 바깥
공백만 제거한다. `번역:`, `요약:`, `답변:`, 추가 설명, reasoning/control token,
prompt echo 또는 명백한 markdown framing이 하나라도 검출되면 평가와 W&B
업로드를 중단한다. 사료 본문 자체의 번호 목록이나 `비고:`는 보존한다. 점수를
높이기 위한 출력 정리나 label 제거는 하지 않는다.

## Data contract

원본 split은 아래 read-only 경로를 그대로 따른다.

`/data/home/mk01/NIPAHistoryTransSummary/output/share_api_hardterm_datasets_20260712/combined_dedup/hard_subset/data`

학습 입력은 이미 split leakage와 human-gold group 제외를 검증한 mk03-owned
prepared data를 읽는다.

`/data/home/mk03/workspace/model_training/modern_korean_lm/artifacts/nodict_official_20260719/data`

Human gold 원본:

`/data/home/mk03/workspace/human-test/data/test-human-infer-vocab-final.jsonl`

평가 prompt와 metric 구현은 아래 원본을 byte-for-byte 복사해 checksum으로
고정한다.

`/data/home/mk01/NIPAHistoryTransSummary/scripts/shared_nodict_evaluation.py`

평가지표는 원본 코드와 동일한 corpus BLEU, sentence BLEU mean, ROUGE-L F1,
BERTScore F1이다. Full test와 human gold는 서로 다른 표와 W&B namespace로
남긴다.

## LoRA scope

`all-linear`는 사용하지 않는다. Solar Open 2의 320 routed experts까지 rank 32
LoRA를 걸면 약 79.5억 adapter parameter가 추가되어 현실적인 optimizer memory를
넘는다. 다음 528개 module만 명시적으로 사용한다.

- 모든 layer의 `q_proj`, `k_proj`, `v_proj`, `o_proj`
- full-attention layer의 `g_proj`
- linear-attention layer의 `b_proj`, `f_a_proj`, `f_b_proj`,
  `g_a_proj`, `g_b_proj`
- 모든 layer의 `mlp.shared_experts.{gate_proj,up_proj,down_proj}`

예상 trainable parameter는 rank 32 기준 133,177,344개다. Routed expert,
router, embedding, norm, lm head는 frozen이다.

## Validation

Static contract:

```bash
python scripts/validate_contract.py
python -m unittest scripts.quantization.test_quantization -v
python -m unittest scripts.test_check_tokenizer_output_contract -v
```

실제 model stage와 3개 GPU가 준비된 후:

```bash
python scripts/preflight_model.py \
  --source-dir /data1/jhtak/Solar-HistoryTransSummary-Open2/hf_cache/models--upstage--Solar-Open2-250B/snapshots/8d1dda5ad2af58207c65348d825772d2f4d72481 \
  --expert-nf4-dir /data1/jhtak/Solar-HistoryTransSummary-Open2/models/Solar-Open2-250B-NF4-layer-flat-experts \
  --download-receipt /data1/jhtak/Solar-HistoryTransSummary-Open2/receipts/bf16_download.json \
  --receipt /data1/jhtak/Solar-HistoryTransSummary-Open2/receipts/model_preflight.json
```

다운로드가 끝나면 `sbatch/00_promote_environment.sbatch`가 자체 포함 환경을
최종 경로로 승격하고 실제 Solar tokenizer로 train/validation 대표와 sealed
combined evaluation 19,778행을 검사한다. Evaluation prompt가 translation
8,192 또는 summarization 10,240 token budget을 넘으면 truncation하지 않고
실패한다. GPU smoke는 `sbatch/03_train_smoke.sbatch`를 사용한다. 이 smoke의
출력은 `gpu_smoke_v7_9600_earlystop`, inference context는 16,384이며 base/tuned
본문-only 출력을 모두 검사한다. 본 학습 template은
`sbatch/04_train_production.sbatch`이고 v7 smoke receipt를 요구하지만, 두 번째
명시적 승인 전에는 제출하지 않는다. 모든 Slurm template은 mk03 job만 다루고
최대 7일 walltime, `batch` partition, `high` QoS, 3 x RTX 6000 GPU와 requeue용
USR1 checkpoint를 사용한다.

승인 후 사용할 새 학습 output은
`train_2epoch_earlystop_step18631_len9600_v2`, W&B run ID는
`solar_open2_hardterm_2epoch_earlystop_20260727_v2`다. 이름의 `step18631`은
고정 maximum horizon 식별자이며 실제 terminal step은 manifest의 조기 종료
evidence로 구분한다.

학습 완료와 `best_adapter` 선택이 모두 봉인된 뒤의 평가 entrypoint는
`sbatch/05_eval.sbatch`다. 이 template은 base와 validation-best tuned adapter를
동일한 16,384-token context와 deterministic decoding으로 full test 및 human
gold에 각각 추론하고, 고정 reference의 BLEU, ROUGE-L, BERTScore 구현으로 두
split을 분리 리포팅한다. `final_adapter`는 실제 종료 상태의 증거일 뿐 tuned
평가 입력으로 사용하지 않는다.

대용량 model/cache/checkpoint/output은 새 전용 root
`/data1/jhtak/Solar-HistoryTransSummary-Open2` 아래에서만 쓴다. `/data1/jhtak`의
기존 다른 파일이나 디렉터리는 수정하지 않는다.

## Environment

학습·추론·평가는 모두 프로젝트 안의 단일 가상환경
`solar-open-2`를 사용한다. 현재 환경은 Upstage Solar Open 2 Transformers
branch, bitsandbytes 0.49.2, 평가용 BERTScore와 한국어 MeCab을 함께 고정한다.
재현용 설치 명세와 freeze 결과는 `requirements/`에 있다. 이 환경은 별도
Qwen/system site-packages에 의존하지 않는 self-contained prefix이며, `/data1`에는
가상환경을 두지 않는다.

실제 환경 경로:

`/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/solar-open-2`

## Official references

- Upstage 소개: <https://www.upstage.ai/blog/ko/solar-open-2?page_slug=ko%2Fsolar-open-2>
- Hugging Face weight: <https://huggingface.co/upstage/Solar-Open2-250B>
- Technical report: <https://arxiv.org/abs/2607.20062>

라이선스 원문은 [LICENSE](LICENSE), attribution은 [NOTICE](NOTICE), 판단 근거와
재현 메모는 [docs/LESSONS_LEARNED.md](docs/LESSONS_LEARNED.md)에 있다.
