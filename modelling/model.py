from torch import nn
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset

import pandas as pd, numpy as np


class SynsetClassificationModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(SynsetClassificationModel, self).__init__()
        self.mlmodel = AutoModel.from_pretrained(model_name)

        self.hidden_size = self.mlmodel.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_classes),
        )

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

        self.num_classes = num_classes

    def forward(self, **inputs):
        masked_labels = inputs.pop("labels")
        sense_labels = inputs.pop("sense-labels")

        sense_labels[masked_labels == -100] = -100
        # Execute masking model
        transformer_output = self.mlmodel(**inputs)
        hidden_state = transformer_output.last_hidden_state

        # masked_word_idx = (labels != -100)
        classifier_logits = self.classifier(hidden_state.view(-1, self.hidden_size))

        loss = self.loss(
            classifier_logits.view(-1, self.num_classes), sense_labels.view(-1)
        )

        return TokenClassifierOutput(loss=loss, logits=classifier_logits)
