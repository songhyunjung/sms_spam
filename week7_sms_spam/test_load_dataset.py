from transformers import BertTokenizer
from data.sms_spam_dataset import load_sms_spam_dataset

def main():
    # BERT 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # 데이터셋 로드 및 토크나이즈
    train_ds, test_ds = load_sms_spam_dataset(tokenizer)

    # 데이터 정보 출력
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"첫번째 훈련 샘플 input_ids 길이: {len(train_ds[0]['input_ids'])}")
    print(f"첫번째 훈련 샘플 라벨: {train_ds[0]['label']}")
    print(f"첫번째 훈련 샘플 텍스트 예시: {tokenizer.decode(train_ds[0]['input_ids'], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
