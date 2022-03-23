from torch import nn
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset

import pandas as pd, numpy as np

class SynsetClassificationModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int
    ):
        super(SynsetClassificationModel, self).__init__()
        self.mlmodel = AutoModel.from_pretrained(model_name)

        self.hidden_size = self.mlmodel.config.hidden_size
        self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.hidden_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, num_classes)
        )

        self.loss = torch.nn.CrossEntropyLoss()

        self.num_classes = num_classes

    def forward(self, **inputs):
        labels = inputs.pop("labels")
        sentence_idx = inputs.pop("sentence_idx")
        # Execute masking model
        transformer_output = self.mlmodel(**inputs)
        hidden_state = transformer_output.last_hidden_state
        # todo : support multiple masks in one sentence
        masked_word_idx = (labels != -100)
        classifier_logits = self.classifier(hidden_state[:, masked_word_idx, :].view(-1, self.hidden_size))

        targets = labels[:, masked_word_idx].unsqueeze(dim=-1)
        loss = self.loss(classifier_logits, targets)

        return TokenClassifierOutput(loss=loss, logits=classifier_logits)

        sentence_idx = inputs.pop("sentence_idx")
        sentence_idx_df = pd.DataFrame({"sentence_idx": sentence_idx})
        lossable = pd.merge(
            self.semcor_dataset.masked, sentence_idx_df, how="inner", on="sentence_idx"
        )


        top2_idx = torch.topk(ml_logits, k=2)
        # top2_predictions = 

        unmasked = self.mlunmasker(lossable["masked"].tolist(), top_k=2)
        unmasked_df = pd.concat(
            [
                pd.DataFrame(prediction[0])
                if masked != prediction[0]["token_str"]
                else pd.DataFrame(prediction[1])
                for prediction, masked in zip(unmasked, lossable["token"])
            ]
        )

        # Add sentence idxs to predictions
        matching_idxs = (
            pd.concat([sentence_idx_df, sentence_idx_df])
            .sort_index(kind="mergesort")
            .set_index("index")
        )
        unmasked_df = pd.concat(unmasked_df, matching_idxs, axis=1)
        with_gold_standard = pd.merge(
            unmasked_df, lossable, on="sentence_idx", how="inner"
        )

        # TODO: can this be accomplished using token_id and not need to recall the tokenizer?
        mlpredictions = with_gold_standard.token_str
        tc_ds = Dataset.from_pandas(mlpredictions).map(
            lambda chunk: self.tokenizer(
                chunk.token_str, padding="longest", truncation="longest_first"
            ),
            batched=True,
        )
        tc_output = self.tcmodel(tc_ds)

        loss, logits = tc_output
        assert "labels" in inputs
        #cel = nn.CrossEntropyLoss()(logits.view(-1, self.label_count), labels.view(-1))
        cel = nn.CrossEntropyLoss()(logits.view(-1), labels.view(-1))

        # TODO: Can we do this without the hidden_states and attentions arguments?
        # TODO: Is it OK to return TokenClassifierOutput even though we want to train a Masked Language Model?
        return TokenClassifierOutput(loss=cel, logits=logits)
