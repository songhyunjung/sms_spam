import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data.sms_spam_dataset import load_sms_spam_dataset
from models.bert_classifier import BertClassifier
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

class SpamClassifierModule(pl.LightningModule):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.model = BertClassifier()
        self.lr = lr
        self.accuracy = Accuracy(task="binary")
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 데이터 경로만 변경
    train_ds, val_ds = load_sms_spam_dataset(tokenizer, 
                          train_file="./week7_sms_spam/sms_spam_train.parquet")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = SpamClassifierModule()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else None

    # 체크포인트 콜백 추가: val_loss가 가장 낮을 때 저장
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./lightning_logs/checkpoints',
        filename='spam-classifier-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],  # 체크포인트 콜백 등록
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
