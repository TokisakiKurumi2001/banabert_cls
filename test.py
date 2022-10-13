from banabert_cls import BanaBERTForSeqClassifier, train_dataloader
for batch in train_dataloader:
    break
batch.pop('labels')
model = BanaBERTForSeqClassifier(
    num_classes=5, model_ck="banabert_model/ot_cl", layers_use_from_last=4
)
output = model(batch)
print(output.shape)