import streamlit as st
import torch
from transformers import BertTokenizer
from models.bert_classifier import BertClassifier
from collections import OrderedDict

# GPU 또는 CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델과 토크나이저 초기화 함수
@st.cache_resource  # 캐싱해서 재사용
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertClassifier()
    model.to(device)

    # 체크포인트 로드
    checkpoint = torch.load("/root/lightning_logs/checkpoints/spam-classifier-epoch=02-val_loss=0.0319.ckpt", map_location=device)
    state_dict = checkpoint['state_dict']

    # 'model.' 접두사 제거
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_key = k[len('model.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model, tokenizer

# 입력 텍스트에 대한 예측 함수
def predict_spam(model, tokenizer, device, texts):
    # 토크나이저로 인코딩 (attention_mask 포함)
    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 모델에 입력 (input_ids, attention_mask 둘 다 전달)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
    return preds

def main():
    st.title("SMS Spam Classifier Demo")

    model, tokenizer = load_model_and_tokenizer()

    st.header("예제 5개 문장 예측 결과")
    # 예제 문장, 정답, 예측 데이터를 임의로 넣거나 파일에서 읽어 사용
    example_texts = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "Nah I don't think he goes to usf, he lives around here though",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
        "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free!",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight"
    ]
    example_labels = [1, 0, 1, 1, 0]  # 1=스팸, 0=정상
    example_preds = predict_spam(model, tokenizer, device, example_texts)

    # 결과 테이블 출력
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
