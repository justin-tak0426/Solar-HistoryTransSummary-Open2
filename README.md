# Solar-HistoryTransSummary-Open2

**Built with Solar**

Solar Open2 250B를 기반으로 근대 사료를 현대 한국어로 번역·요약하는 LoRA 모델입니다. 학습 코드, 최종 어댑터, 추론 예제를 제공합니다.

## 설치

Python 3.11, CUDA GPU, Git LFS가 필요합니다.

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

## 추론

UTF-8 텍스트 파일에 사료 본문을 넣고 실행합니다.

```bash
# 번역
python inference.py --task translation --source-file source.txt

# 요약
python inference.py --task summarization --source-file source.txt

# 로컬 백본 사용
python inference.py --base /path/to/base-model --source-file source.txt
```

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `--source-file` | 입력 텍스트 파일 | 필수 |
| `--task` | `translation` 또는 `summarization` | `translation` |
| `--base` | 백본 모델 ID 또는 로컬 경로 | `upstage/Solar-Open2-250B` |
| `--adapter` | LoRA 어댑터 경로 | `adapter/` |
| `--max-new-tokens` | 최대 출력 토큰 수 | 번역 8,192 / 요약 6,144 |

어댑터는 약 509 MiB이며 백본이 별도로 필요합니다. 위 예제는 BF16 백본을 로딩하므로 가중치만 약 500.6 GB이고, 추가 GPU 메모리도 필요합니다. 전체 입력·출력은 16,384 tokens 이내여야 합니다. 예제의 GPU 실행은 별도로 검증하지 않았습니다.

## 학습

학습 코드는 `project/scripts/train.py`, 설정은 `project/config/b200/`에 있습니다.

| 항목 | 설정 |
| --- | --- |
| 백본 | `upstage/Solar-Open2-250B` |
| GPU | NVIDIA B200 ×2 |
| 방식 | Routed expert NF4 양자화 + attention/shared expert LoRA |
| LoRA | rank 32, alpha 64, dropout 0.05 |
| Epoch / Step | 1 / 9,316 |
| 최대 시퀀스 길이 | 9,600 tokens |
| Effective batch size | 16 |
| Learning rate | 1e-4 |

최종 어댑터는 1 epoch 학습을 완료한 체크포인트입니다. 학습 요약은 [training/summary.json](training/summary.json)에 있습니다.

기존 학습 실행 스크립트는 `project/launch/b200/train_production.sh`입니다. 재학습하려면 실행 스크립트와 `project/scripts/b200_runtime.py`, 설정 파일의 경로·환경을 수정하고 학습 데이터 및 NF4 자산·검증 파일을 준비해야 합니다. 학습 데이터와 백본 가중치는 포함하지 않습니다.

## 코드 구성

| 경로 | 용도 |
| --- | --- |
| `inference.py` | BF16 백본과 LoRA를 로딩하는 추론 예제 |
| `adapter/` | 최종 어댑터와 토크나이저 |
| `project/scripts/train.py` | 학습 |
| `project/scripts/infer.py` | 기존 평가용 추론 |
| `project/scripts/quantization/` | NF4 변환·로딩 |
| `project/config/b200/` | 학습 설정 |
| `project/requirements/` | 학습 환경 의존성 |
| `training/` | 학습 결과 기록 |

## 라이선스

[Upstage Solar License](LICENSE.md)를 따릅니다. 출처 및 변경 사항은 [NOTICE](NOTICE)에 명시되어 있습니다. 본 모델은 Upstage의 공식 제품이 아닙니다.
