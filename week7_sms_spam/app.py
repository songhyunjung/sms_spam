import streamlit as st
import torch
from transformers import BertTokenizer
from models.bert_classifier import BertClassifier
from collections import OrderedDict
from huggingface_hub import hf_hub_download

# GPU 또는 CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델과 토크나이저 초기화 함수
@st.cache_resource  # 캐싱해서 재사용
def load_model_and_tokenizer():
    # 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 모델 클래스 초기화 및 device 할당
    model = BertClassifier()
    model.to(device)

    # Hugging Face Hub에서 체크포인트 다운로드
    checkpoint_path = hf_hub_download(
        repo_id="songhyunjung/spam-classifier-model",  # 본인 repo 이름
        filename="spam-classifier-epoch=02-val_loss=0.0319.ckpt"  # 업로드한 체크포인트 파일명
    )

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']

    # 'model.' 접두사 제거
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

    return model, tokenizer

# 입력 텍스트에 대한 예측 함수
def predict_spam(model, tokenizer, device, texts):
    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
    return preds

def main():
    st.title("SMS Spam Classifier Demo")

    model, tokenizer = load_model_and_tokenizer()

    st.header("예제 5개 문장 예측 결과")
    example_texts = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "Nah I don't think he goes to usf, he lives around here though",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
        "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free!",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight"
    ]
    example_labels = [1, 0, 1, 1, 0]  # 1=스팸, 0=정상
    example_preds = predict_spam(model, tokenizer, device, example_texts)

    import pandas as pd
    df_examples = pd.DataFrame({
        "문장": example_texts,
        "정답": example_labels,
        "예측": example_preds
    })
    accuracy = sum([a == b for a, b in zip(example_labels, example_preds)]) / len(example_labels)
    st.write(f"예제 데이터 정확도: {accuracy:.4f}")
    st.dataframe(df_examples)

    st.header("텍스트 입력 후 실시간 스팸 예측")
    user_input = st.text_area("SMS 문장 입력", "")

    if user_input.strip():
        preds = predict_spam(model, tokenizer, device, [user_input])
        pred_label = "스팸" if preds[0] == 1 else "정상"
        st.write(f"예측 결과: **{pred_label}**")

if __name__ == "__main__":
    main()
