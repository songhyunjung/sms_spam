import torch
from torch.utils.data import DataLoader
import pandas as pd
from collections import OrderedDict

from transformers import BertTokenizer
from data.sms_spam_dataset import load_sms_spam_dataset
from models.bert_classifier import BertClassifier


def predict_and_evaluate():
    # 사용할 디바이스 설정 (GPU 있으면 cuda, 없으면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # BERT 기본 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 테스트 데이터셋 로드 (test.parquet 경로 확인 필요)
    test_dataset = load_sms_spam_dataset(tokenizer, test_file="/root/week7_sms_spam/sms_spam_test.parquet")

    # DataLoader 정의 (배치 사이즈 32, shuffle 안함)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 모델 초기화 및 device에 할당
    model = BertClassifier()
    model.to(device)

    # 저장된 체크포인트 불러오기
    checkpoint = torch.load("/root/lightning_logs/checkpoints/spam-classifier-epoch=02-val_loss=0.0319.ckpt", map_location=device)
    state_dict = checkpoint['state_dict']

    # 체크포인트 키 앞에 'model.' 접두사 제거
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_key = k[len('model.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    # 모델에 가중치 로드
    model.load_state_dict(new_state_dict)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 모델 출력 (attention_mask를 사용하는 경우)
            outputs = model(inputs, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 정확도 계산
    accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 예측 결과 DataFrame으로 저장
    df = pd.DataFrame({
        'label': all_labels,
        'prediction': all_preds
    })

    df.to_csv("prediction_results.csv", index=False)
    print("Prediction results saved to prediction_results.csv")


if __name__ == "__main__":
    predict_and_evaluate()
