# Lessons Learned: Solar Open 2 역사 번역·요약 튜닝

**Built with Solar**

> Historical v1 baseline: the 4,000-step/3,072-token details below document the
> first design and must not be used as current execution instructions. The active
> two-epoch/9,600-token contract is
> [APPROVED_TRAINING_DESIGN_20260727.md](APPROVED_TRAINING_DESIGN_20260727.md).
> No smoke-dependent production job may be pre-submitted before its required
> second explicit approval.

## 1. 목표와 고정 조건

이 실험은 기존 Qwen 번역·요약 실험에서 효과가 확인된 학습 규칙을 Solar Open 2에
이식한다. 비교 가능한 조건은 4,000 optimizer step, rank 32, effective batch 16,
3,072 token, learning rate 1e-4, 목표 독음 2.5배 loss, 전문가 교정 8배
oversampling, NEFTune alpha 5, 번역/요약 sampling 0.65/0.35다.

마지막 step 4,000을 사용한다는 조건도 고정한다. Validation은 학습 이상과
overfitting을 감시하지만 더 이른 checkpoint를 고르는 용도로 쓰지 않는다.

## 2. 데이터 split에서 얻은 교훈

Train, validation, test의 이름만 다르다고 leakage가 없는 것은 아니다. 이
프로젝트는 source-declared split을 유지하면서 human-gold group을 train과
validation에서 제외한 prepared data를 사용한다. Exact `(id, task)` 및 정규화한
`(origin, task)` overlap이 0인 manifest를 checksum으로 고정했다.

Validation을 생략하면 loss curve만 보고 학습이 정상인지 판단하게 된다. 이번
설계는 500 step마다 고정 subset, 마지막에 전체 eligible validation을 측정한다.
Test와 human gold는 마지막 adapter가 확정될 때까지 열지 않는다.

## 3. 어려운 용어 loss 2.5배

전체 문장 CE만 줄이면 흔한 연결어와 문체 개선이 loss를 지배한다. 평가에서 중요한
어려운 용어의 목표 독음 token은 상대적으로 드물다. 따라서 target response 안에서
`hard_term_readings`의 정확한 token subsequence를 찾고 해당 token CE만 2.5배로
가중한다.

안전 규칙은 다음과 같다.

- Prompt token은 항상 `-100`으로 mask한다.
- Response에서 정확히 정렬된 reading token만 가중한다.
- 한 reading이 여러 번 나오면 모든 정확한 occurrence를 가중한다.
- 정렬이 불명확하면 가중하지 않는다.
- Validation loss는 1.0배 CE로 계산해 모델 간 비교 가능성을 유지한다.

## 4. 전문가 교정 8배 oversampling

현재 train 142,713건 중 `gold_expert_corrected`는 905건뿐이다. 단순 무작위
sampling에서는 전문가 교정 신호가 optimizer update에 거의 등장하지 않는다.
Task별 pool에 전문가 row를 7회 추가해 총 8배 노출하고, 각 task 안에서는
without-replacement shuffled deck을 사용한다.

Oversampling은 새로운 사실을 추가하지 않는다. 같은 예시의 반복이므로 지나친
배수는 문장 암기 위험을 높인다. 8배는 희소 신호를 보강하는 고정 실험값이고,
validation 및 human gold 결과로 사후 검증해야 한다.

## 5. 3,072 token과 effective batch 16

2,048 token은 긴 사료를 더 많이 버린다. 3,072는 coverage와 memory 사이의
절충이다. Micro batch 1, accumulation 16으로 effective batch 16을 만든다.
`effective batch = micro batch x accumulation x data-parallel world size`인데, 이번
구성은 DDP가 아니라 model parallel single process이므로 world size는 1이다.

## 6. 왜 Solar Open 2에서 DDP를 쓰지 않는가

Solar Open 2는 총 250B parameter, token당 약 15B active인 MoE다. MoE의 active
parameter가 작아도 weight 저장 공간은 총 parameter에 의해 결정된다. BF16
weight가 약 500.6 GB라서 3 x 96 GB에 들어가지 않으며, DDP는 각 GPU에 model을
복제하므로 더 불가능하다.

학습은 공식 BF16 fixed revision을 source of truth로 유지한다. 241.6B routed
expert parameter만 audited bitsandbytes NF4 sidecar로 변환하고 나머지 weight는
같은 BF16 snapshot에서 직접 적재한다. 긴 sequence 학습은 layer-flat chunk
layout, autoregressive 평가 추론은 per-expert layout을 사용한다. 두 artifact는
동일 source index/config hash에 묶인다. Nota NVFP4는 deployment-oriented
inference 후보일 뿐 학습 기본값이 아니다. Custom NF4 kernel의 input activation
backward는 반드시 1-update smoke로 확인한다.

## 7. 왜 all-linear LoRA를 버렸는가

Solar Open 2는 layer마다 320 routed expert를 갖는다. 모든 expert projection에
rank 32 LoRA를 추가하면 약 7.95B trainable parameter가 생기고 optimizer state가
GPU memory를 압도한다.

대신 attention과 항상 활성화되는 shared expert에만 LoRA를 건다. 고정 architecture
기준 528개 module, 133,177,344 trainable parameter다. Router와 320 routed expert
weight는 frozen이다. LoRA 대상은 공식 BF16 snapshot에서 그대로 적재하고
routed expert만 NF4로 바꾼다.

이 변경은 Qwen `all-linear`와 완전히 동일한 intervention이 아니다. 결과
리포트에서 architecture adaptation으로 명시해야 하며, 모델 자체 효과와 LoRA
scope 효과를 혼동하면 안 된다.

## 8. NEFTune alpha 5

Training 중 input embedding에
`alpha / sqrt(sequence_length * hidden_size)` 크기의 uniform noise를 더한다.
Inference와 validation model eval mode에서는 noise를 넣지 않는다. 이 방식은
추가 parameter 없이 instruction tuning의 국소적인 과적합을 완화한다.

## 9. Prompt와 reasoning mode

학습과 평가 모두 reference evaluator의 `build_messages_no_dict`를 사용한다.
Solar Open 2 chat template의 기본 reasoning mode는 high이므로, 호출할 때마다
`reasoning_effort="none"`과 `think_render_option="preserved"`를 명시한다. 이를
빠뜨리면 reasoning token이 길어지고 본문-only 평가 계약을 깨뜨릴 수 있다.

학습 prompt token은 label `-100`, loss weight 0으로 완전히 mask하고 assistant의
번역·요약 본문과 정확히 하나의 `<|im:end|>`만 감독한다. 이 token ID가 공식
`generation_config.json`의 EOS 집합에 포함되는지 확인하며, template의 후행
줄바꿈이나 다른 suffix는 loss에 넣지 않는다. 원본 train에 `답변:`이 붙은
것으로 고정 확인된 3건은 학습 타깃에서 그 prefix만 제거한다. 반면 sealed
full-test reference 3건은 evaluator 동일성을 위해 수정하지 않는다.

평가 출력은 후처리로 고치지 않는다. EOS/padding까지만 token 단계에서 종료하고
special token을 숨기지 않은 raw continuation의 바깥 whitespace만 제거한다.
Label, 추가 설명 framing, reasoning/chat-control token, prompt echo 또는 명백한
markdown framing이 한 건이라도 검출되면 metric 계산과 보고를 중단한다. 사료
본문 자체의 번호 목록과 `비고:`는 output body로 보존한다.

## 10. 평가와 보고

학습 전 Solar base와 step-4000 adapter를 동일 prompt, decoding, metric으로
평가한다. Full test 18,557건과 human gold 1,221건은 분리한다. 각 split에서
translation/summarization별로 다음을 남긴다.

- Corpus BLEU
- Sentence BLEU mean
- ROUGE-L F1 mean
- BERTScore F1 mean (`xlm-roberta-large`)
- 학습 전 base 대비 delta

W&B에는 train loss, learning rate, validation history, 두 split의 절대 점수와
base 대비 delta, 최종 report artifact를 기록한다. 학습과 평가는 서로 다른 W&B
run을 사용해 같은 step 번호의 history가 덮이거나 drop되지 않게 하고, 평가
artifact에는 manifest로 봉인한 실제 학습 entity/project/run id와 audited
prediction/metric checksum을 함께 남긴다.

## 11. 중단 복구와 운영 원칙

- Slurm walltime은 가능한 최대인 7일을 사용한다.
- 종료 2시간 전 SIGUSR1을 받아 optimizer update 경계에서 저장한다.
- Accumulation 도중 신호가 오면 partial update의 gradient, sampler와 RNG 진행을
  버리고 직전 optimizer 경계로 되돌린 뒤 저장한다.
- Adapter, optimizer, RNG, sampler deck과 완료 optimizer step을 함께 저장한다.
- Scheduler는 mutable state를 저장하지 않고 durable `completed_steps`로 다음
  update의 learning rate를 재계산한다. 현재 2-epoch 설계는 peak `1e-4`,
  558-update warmup 뒤 cosine decay, minimum ratio 0.1(`1e-5` floor)이며 모든
  data-bearing update의 learning rate가 양수여야 한다.
- 재시작 시 model/data/config, 실행 코드 tree와 environment lock checksum이
  다르면 resume을 거부한다.
- 현재 `mk03` 사용자와 현재 job id의 소유권을 확인한 뒤 자기 job만 requeue한다.
- 다른 사용자의 파일과 job은 읽기 reference 외에는 변경하지 않는다.
- 대용량 산출물은 새 전용 root
  `/data1/jhtak/Solar-HistoryTransSummary-Open2` 아래에서만 생성한다.
- GPU smoke에서 3,072-token microbatch와 optimizer boundary 시간을 따로 측정한다.
  64,000 production microbatch, 4,000 optimizer boundary, 7회의 1,024-row
  validation과 마지막 17,850-row validation을 보수적으로 투영해 총 ETA와 필요한
  7일 allocation 수를 receipt에 기록한다.
- USR1을 받은 batch shell은 Python이 optimizer 경계 checkpoint를 완전히 쓰고
  실제 종료할 때까지 기다린 뒤 자기 job만 requeue한다.
- 실제 Solar tokenizer로 combined evaluation 19,778행을 전수 검사해 평가 prompt
  truncation이 0건인지 확인하고 이 receipt를 production smoke에 포함한다.

## 12. 아직 실험으로 확인해야 하는 것

1. Upstage Transformers fork + custom bitsandbytes NF4 expert가 input
   activation backward를 지원하는가.
2. PEFT가 지정한 528개 BF16 linear module을 정확히 감싸는가.
3. 3,072 token one-update가 3 x 96 GB에서 CPU/disk offload 없이 통과하는가.
4. Custom NF4 loader의 Transformers generation이 base와 LoRA 모두 동일 prompt로
   완주하는가.
5. Step time으로 계산한 4,000-step 예상 시간이 7일 walltime/requeue 정책 안에서
   감당 가능한가.

이 다섯 항목은 v1 GPU smoke의 역사적 gate다. 현재 v2 학습에는 새 9,600-token
smoke와 [승인 학습 설계](APPROVED_TRAINING_DESIGN_20260727.md)의 전체 gate를
적용한다. 두 번째 명시적 승인 전에는 production job을 dependency 형태로도 미리
제출하지 않는다.
