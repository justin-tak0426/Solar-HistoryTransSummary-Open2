# Solar-HistoryTransSummary-Open2-B200

**Built with Solar**

근대 동아시아 사료를 현대 한국어로 번역·요약하도록 `upstage/Solar-Open2-250B`를 LoRA fine-tuning한 결과와 실제 학습 코드입니다. 사전·용어집을 입력하지 않는 프롬프트를 사용합니다.

## 모델과 결과물

| 항목 | 값 |
| --- | --- |
| 백본 | `upstage/Solar-Open2-250B` |
| 고정 revision | `8d1dda5ad2af58207c65348d825772d2f4d72481` |
| 아키텍처 | `SolarOpen2ForCausalLM`, MoE, 전체 약 250B / 토큰당 활성 약 15B |
| 저장 형식 | PEFT LoRA adapter, 약 509 MiB |
| 최종 학습 | 1 epoch, 9,316 optimizer updates, 조기 종료 없음 |
| 최종 선택 | step 9,316; best와 final의 가중치가 동일 |

`adapter/adapter_model.safetensors`가 최종 가중치입니다. 백본 전체를 병합한 모델이 아니므로 로딩할 때 원본 백본도 필요합니다. 중복 저장을 피하기 위해 best/final 중 동일한 가중치 한 벌만 포함했습니다.

검증 selection loss는 `0.65 × translation_loss + 0.35 × summarization_loss`입니다.

| Step | Epoch | Translation loss | Summarization loss | Selection loss |
| --- | --- | --- | --- | --- |
| 4,658 | 약 0.5 | 0.420805 | 0.650390 | 0.501160 |
| 9,316 | 1.0 | 0.392234 | 0.618610 | **0.471465** |

위 수치는 학습 중 validation 결과이며 test 성능 수치가 아닙니다. 근거는 [학습 요약](training/summary.json)과 [원본 실행 기록](training/run_manifest.json)에 있습니다.

## 파일 구성

- `adapter/`: 최종 LoRA 가중치, tokenizer, chat template 및 원본 모델 카드
- `inference.py`: 다른 서버용 BF16 백본 + LoRA 로딩·추론 예제
- `requirements-inference.txt`: 예제용 Solar/PEFT 의존성
- `project/scripts/train.py`: 실제 학습 코드
- `project/scripts/solar_inference_loader.py`: 기존 2×B200 NF4 추론 로더
- `project/scripts/quantization/`: NF4 변환 및 로더
- `project/config/b200/`: 완료된 1-epoch 학습의 설정
- `project/launch/b200/`: 원래 서버의 학습·평가 실행 스크립트
- `project/requirements/solar-open-2.lock.txt`: 보관된 학습 환경 패키지 목록
- `project/README.original.md`, `project/docs/`: 이전 설계 및 원본 문서

이전 문서에는 다른 서버의 2-epoch 설계도 남아 있습니다. **완료된 이 모델의 학습 사실은 `training/run_manifest.json`과 `project/config/b200/`를 기준으로 확인하세요.**

## 다운로드와 설치

가중치는 Git LFS로 관리합니다. 저장소를 clone하기 전에 Git LFS를 설치한 뒤 실행하세요.

```bash
git lfs install
git clone https://github.com/justin-tak0426/Solar-HistoryTransSummary-Open2.git
cd Solar-HistoryTransSummary-Open2
git lfs pull
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
```

학습 계약의 핵심 버전은 Python 3.11, PyTorch 2.11.0+cu128, Upstage Transformers commit `ef5c37485091057070b36dd0620798f74988eb01`, PEFT 0.19.1, bitsandbytes 0.49.2, Accelerate 1.14.0, fla-core 0.5.1입니다. 일반 Transformers 배포판으로 임의 교체하지 마세요. 원본 lock은 학습 당시 환경 기록이며 다른 서버의 CUDA 드라이버·GPU에 맞는 설치 확인이 필요합니다.

## 다른 서버에서 BF16으로 로딩·추론

```bash
python inference.py --task translation --source-file source.txt
python inference.py --task summarization --source-file source.txt
# 동일한 고정 revision의 로컬 BF16 snapshot 사용
python inference.py --base /path/to/pinned/snapshot --source-file source.txt
```

`source.txt`에 UTF-8 사료 본문을 넣습니다. `--base`를 생략하면 고정 revision의 공식 백본을 다운로드합니다. 백본 BF16 가중치만 약 **500.6 GB**이므로 그 이상의 합산 GPU 메모리와 activation/KV cache 여유, 다운로드 저장 공간이 필요합니다. 이 예제는 원래 2×B200 NF4 실행 방식과 메모리 요구가 다릅니다. `device_map="auto"`의 배치는 가용 메모리에 따라 달라지며 CPU/disk 배치 시 매우 느릴 수 있습니다.

예제는 학습 당시 번역·요약 few-shot 프롬프트와 adapter의 chat template를 사용합니다. reasoning은 끄고, deterministic decoding과 `<|im:end|>` 종료 토큰을 사용합니다. 전체 context budget은 16,384 tokens이며 입력을 자동으로 자르지 않습니다. 기본 최대 출력은 번역 8,192 / 요약 6,144 tokens입니다. `--max-new-tokens`로 조절할 수 있습니다.

PEFT 0.19.1의 native Solar Open2 로딩 호환성 처리를 위해 `project/scripts/solar_peft_compat.py`의 context manager 안에서 adapter를 로딩합니다. 제공한 예제에 해당 처리가 포함되어 있습니다.

**검증 범위:** 이 공개용 BF16 예제는 구문·CLI 확인 대상이며, 이 패키지를 준비하면서 250B GPU 로딩·생성을 새로 실행하지는 않았습니다. 원래 실행 결과는 아래의 custom NF4 환경에서 생성되었습니다.

## 원래 2×B200 NF4 환경에서 추론

학습에서는 공식 BF16 snapshot의 routed expert만 bitsandbytes NF4로 변환했습니다. Attention, shared expert 등은 BF16이며 LoRA는 attention/shared expert projection에 적용됩니다. 전체 모델을 일반적인 `load_in_4bit=True`로 로딩한 방식과 다릅니다.

원래 실행에는 다음 자산이 필요합니다.

1. 위 고정 revision의 공식 BF16 snapshot
2. 학습용 `Solar-Open2-250B-NF4-layer-flat-experts` sidecar
3. 추론용 `Solar-Open2-250B-NF4-per-expert-fallback` sidecar
4. 최종 adapter와 tokenizer
5. 원본 pipeline이 요구하는 asset integrity / preflight / smoke / terminal training 증빙

학습용·추론용 sidecar와 500GB 원본 백본은 이 저장소에 포함하지 않습니다. 변환 코드와 계약은 `project/scripts/quantization/`에 있습니다. 기존 변환은 별도 GPU capability 계약이 있으므로 B200에서 무조건 재생성할 수 있다고 가정하면 안 됩니다.

`project/scripts/infer.py`는 봉인된 평가 데이터와 증빙을 검증하는 기존 평가 CLI입니다. 임의 텍스트용 범용 CLI가 아닙니다. 실제 실행 흐름은 `project/launch/b200/run_post_training_evaluation.sh`와 `project/scripts/solar_inference_loader.py`를 참고하세요. `nota-ai/Solar-Open2-250B-Nota-NVFP4`는 별도 추론 후보이며 이 adapter를 학습한 백본은 공식 Upstage BF16 모델입니다.

## 학습 내용과 재현

| 항목 | 값 |
| --- | --- |
| Hardware | NVIDIA B200 ×2, single-process model parallel |
| LoRA | rank 32, alpha 64, dropout 0.05 |
| 학습 파라미터 | 133,177,344, 총 528개 대상 module |
| 학습 대상 | attention projections 및 shared-expert gate/up/down projections |
| Frozen | routed experts, router, embedding, norm, lm_head |
| 최대 시퀀스 길이 | 9,600; 실제 최대 9,560 tokens |
| Effective batch | 16 occurrences; 마지막 step은 8 |
| Learning rate | peak 1e-4, 279-step warmup, cosine floor 1e-5 |
| Weight decay / clipping | 0.01 / 1.0 |
| Seed / NEFTune alpha | 42 / 5 |
| 난해 용어 target reading loss | 2.5배 |
| 전문가 교정 데이터 | epoch당 8회 노출 |

원본 unique training rows는 142,713개입니다. 전문가 교정 905개를 8회 노출하여 총 149,048 training occurrences(번역 102,224 / 요약 46,824)를 한 epoch 동안 처리했습니다. 누락된 occurrence는 0개입니다. Prompt token은 loss에서 제외하고 assistant 답변 및 종료 토큰만 학습합니다. Test/human gold는 checkpoint 선정에 사용하지 않습니다.

원래 학습 진입점은 다음입니다.

```bash
bash project/launch/b200/train_production.sh
```

**위 명령은 원래 서버 환경 보존용입니다. 새 서버에서 바로 실행되는 범용 명령이 아닙니다.** Launcher와 Python runtime이 원래 절대 경로, 사용자, GPU 6·7, 환경 prefix, checksum 및 receipt를 검사합니다. 다른 환경에서는 `project/scripts/b200_runtime.py`, `project/launch/b200/common_env.sh`, `project/config/b200/`의 경로·하드웨어 계약과 증빙 생성 흐름을 함께 이전해야 합니다. 단순히 JSON 경로만 바꾸면 재현되지 않습니다.

학습 데이터 원문과 평가 데이터는 포함하지 않았습니다. 동일 학습 재현에는 원본 데이터 및 preprocessing 계약도 필요합니다. 모델 추론에는 학습 데이터가 필요하지 않습니다.

## GitHub에 올리기

Git LFS가 설치된 환경에서 저장소 루트에서 실행합니다.

```bash
git init
git lfs install
git add .
git commit -m "Publish Solar Open2 LoRA adapter and training code"
git branch -M main
git remote add origin https://github.com/justin-tak0426/Solar-HistoryTransSummary-Open2.git
git push -u origin main
```

`.gitattributes`가 `*.safetensors`를 LFS 대상으로 지정합니다. **Git LFS 설치 없이 가중치를 일반 Git 객체로 커밋하지 마세요.** 원본 backbone/cache/venv/dataset은 포함하지 않습니다.

## 라이선스

이 저장소의 모델 어댑터와 자체 작성 코드·설정·문서는 백본과 동일한 **Upstage Solar License**로 배포합니다. 포함된 제3자 코드 및 의존성의 기존 라이선스·저작권 표시는 그대로 적용됩니다.

- [LICENSE](LICENSE): 학습에 사용한 고정 revision의 공식 라이선스 원문을 수정 없이 포함했습니다.
- [NOTICE](NOTICE): 백본 출처·revision, 파생 모델명, LoRA 변경 내용을 명시했습니다.
- `adapter/`에도 LICENSE와 NOTICE를 포함하여 어댑터만 배포해도 라이선스가 함께 전달되도록 했습니다.
- 파생 모델명은 `Solar`로 시작하며, 관련 문서·웹사이트·UI에 **Built with Solar**를 표시해야 합니다(4(e)). 재배포 시 라이선스 사본과 해당되는 원본 고지를 함께 보존해야 합니다.
- 본 모델은 Upstage의 공식 제품이나 보증·인증을 받은 모델이 아닙니다.

공식 출처: [Solar Open2 250B LICENSE (학습 revision)](https://huggingface.co/upstage/Solar-Open2-250B/blob/8d1dda5ad2af58207c65348d825772d2f4d72481/LICENSE).

GitHub 저장소 생성 화면에서는 **Choose a license → None**으로 두고, 이 저장소의 LICENSE 파일을 업로드하세요. Upstage Solar License는 Apache 2.0에 별도 조건을 추가한 라이선스이므로 GitHub의 Apache 2.0 템플릿으로 대체하지 않습니다.
