import torch
from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_classes=2):
        super(BertClassifier, self).__init__()
        # 사전학습된 BERT 모델 불러오기
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        # BERT 출력 차원
        hidden_size = self.bert.config.hidden_size
        # 분류를 위한 선형 레이어 (hidden_size -> num_classes)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # BERT 인코더 통과
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [batch_size, hidden_size] 크기의 pooled output 사용 (CLS 토큰 출력)
        pooled_output = outputs.pooler_output
        # 분류기 통과
        logits = self.classifier(pooled_output)
        return logits
