import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class SynsetClassificationModel(PreTrainedModel):
    def __init__(self, config, model_name: str, num_classes: int, freeze_lm: bool = False):
        super(SynsetClassificationModel, self).__init__(
            config
        )
        self.mlmodel = AutoModel.from_pretrained(model_name)

        if freeze_lm:
            for parameter in self.mlmodel.parameters():
                parameter.requires_grad = False

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
