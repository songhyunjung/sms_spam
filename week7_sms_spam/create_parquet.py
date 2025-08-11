from datasets import load_dataset
import pandas as pd
import os

def save_parquet():
    # Huggingface sms_spam 데이터셋 로드
    dataset = load_dataset("ucirvine/sms_spam")

    # train 데이터셋 DataFrame 변환
    df = pd.DataFrame(dataset['train'])

    # train/test 분할 (예: 80% train, 20% test)
    train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    test_df = df.drop(train_df.index).reset_index(drop=True)

    # 저장할 디렉토리 생성
    os.makedirs("./data", exist_ok=True)

    # parquet 파일로 저장
    train_df.to_parquet("./data/sms_spam_train.parquet", index=False)
    test_df.to_parquet("./data/sms_spam_test.parquet", index=False)

    print("parquet 파일 생성 완료!")

if __name__ == "__main__":
    save_parquet()
