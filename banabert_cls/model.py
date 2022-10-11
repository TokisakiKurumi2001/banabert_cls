import torch
import torch.nn as nn
from transformers import BertModel

class BanaBERTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_classes):
        super(BanaBERTClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class BanaBERTForSeqClassifier(nn.Module):
    def __init__(self, num_classes: int, model_ck: str):
        super(BanaBERTForSeqClassifier, self).__init__()
        self.banabert = BertModel.from_pretrained(model_ck)
        self.classifier = BanaBERTClassificationHead(self.banabert.config, num_classes)
    
    def forward(self, inputs):
        outputs = self.banabert(**inputs)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits