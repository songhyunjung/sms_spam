# SMS Spam Classifier

이 프로젝트는 SMS 메시지를 **스팸 / 정상**으로 분류하는 BERT 기반 분류 모델입니다.  

## 📌 프로젝트 개요

- 목적: SMS 메시지의 스팸 여부를 자동으로 분류하여 사용자 보호
- 모델: PyTorch 기반 **BERT Classifier**
- 데이터셋: SMS Spam Collection (train/test 분할)
- 주요 성능 지표(Test set 기준):
  - Accuracy: 0.9964
  - Precision: 0.9930
  - Recall: 0.9793
  - F1-score: 0.9861

---

## ⚙️ 학습 및 평가 설정

- **Train Batch size**: 16
- **Test Batch size**: 32
- Learning rate: 2e-5
- Epochs: 3
- Optimizer: AdamW
- Scheduler: Linear Warmup Scheduler

---

## 📊 데이터 분석

- SMS 메시지는 일반 텍스트 형태이며, 스팸 메시지는 보통 금전, 프로모션, 링크 등 특정 패턴을 포함
- 데이터 불균형은 학습 과정에서 클래스 가중치 또는 샘플링으로 대응

---

## 🧠 모델 선정 기준

- **BERT**를 선택한 이유:
  - 문맥을 이해하는 능력이 뛰어나 SMS 문장 단위 분류에 적합
  - 사전 학습된 모델을 활용하면 소규모 데이터에서도 높은 성능 가능
- 단순한 로지스틱 회귀/MLP 대비 문맥 기반 분류 성능이 우수

---

## 🛠 실험 셋팅

- 모델 학습 후 체크포인트 저장: `lightning_logs/checkpoints/spam-classifier-epoch=02-val_loss=0.0319.ckpt`
- 테스트 데이터에 대해 모델 평가 및 결과 저장
- 평가 지표: Accuracy, Precision, Recall, F1-score

---

## 📈 정량/정성 평가

- 정량적: Test Accuracy: 0.9964, Precision: 0.9930, Recall: 0.9793, F1-score: 0.9861
- 정성적:
  - 스팸으로 명확히 분류되는 메시지: 금전, 프로모션, 링크 포함
  - 헷갈리는 메시지: 짧거나 모호한 표현, 약간의 정상 메시지와 유사한 문장

---

## 🔍 실험 결과 분석

- 대부분 메시지에서 높은 정확도 달성
- F1-score가 Recall보다 약간 높은 이유: 스팸을 정상으로 잘못 분류하는 경우가 일부 존재
- 헷갈리는 메시지는 짧은 텍스트나 문맥이 모호한 문장

---

## 🚀 향후 연구

- 모델 경량화 및 모바일 배포
- 추가 SMS 데이터 확보 및 다양한 언어 적용
- 모델의 헷갈리는 메시지에 대한 성능 향상 (예: 추가 전처리, 앙상블 모델)
