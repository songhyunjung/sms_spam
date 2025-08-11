import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# PyTorch Dataset 정의
class SmsSpamDataset(Dataset):
    # encodings: tokenizer 결과 (input_ids, attention_mask 등)
    # labels: 스팸 여부 (0, 1)
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # 토크나이저 출력값을 tensor로 변환
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 라벨도 tensor 변환
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 데이터 로드 함수
def load_sms_spam_dataset(tokenizer, train_file=None, test_file=None):
    """
    train_file만 주면 → (train_dataset, val_dataset) 반환
    test_file만 주면 → test_dataset 반환
    """
    if test_file:
        # 테스트 데이터 로드
        df = pd.read_parquet(test_file)
        texts = df['sms'].tolist()
        labels = df['label'].tolist()

        encodings = tokenizer(texts, truncation=True, padding=True)
        dataset = SmsSpamDataset(encodings, labels)
        return dataset

    if train_file:
        # 학습 데이터 로드 후 train/val 분리
        df = pd.read_parquet(train_file)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['sms'], df['label'], test_size=0.2, random_state=42
        )

        train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
        val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

        train_dataset = SmsSpamDataset(train_encodings, train_labels.tolist())
        val_dataset = SmsSpamDataset(val_encodings, val_labels.tolist())

        return train_dataset, val_dataset

    raise ValueError("train_file 또는 test_file 중 하나는 반드시 지정해야 합니다.")
