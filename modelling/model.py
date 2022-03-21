from torch import nn
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset

import pandas as pd, numpy as np

from dataset.datagen import SemCorDataSet


class MaskedLMWithSynsetClassification(nn.Module):
    def __init__(
        self,
        mlmodel: PreTrainedModel,
        num_classes: int,
        tokenizer: PreTrainedTokenizer,
        semcor_dataset: SemCorDataSet,
    ):
        super(MaskedLMWithSynsetClassification, self).__init__()
        self.mlmodel = mlmodel
        self.tokenizer = tokenizer

        self.classifier = torch.nn.Sequential(
            [
                torch.nn.Linear(len(self.tokenizer), 1024),
                torch.nn.Linear(1024, num_classes)
            ]
        )

        self.loss = torch.nn.CrossEntropyLoss()

        # self.dropout = nn.Dropout(0.1)
        self.num_classes = num_classes
        self.semcor_dataset = semcor_dataset

        """ self.mlunmasker = pipeline(
            task="fill-mask", model=self.model, tokenizer=self.tokenizer
        ) """

    def forward(self, **inputs):
        sentence_idx = inputs.pop("sentence_idx")
        sentence_idx_df = pd.DataFrame({"sentence_idx": sentence_idx})
        lossable = pd.merge(
            self.semcor_dataset.masked, sentence_idx_df, how="inner", on="sentence_idx"
        )

        

        # Execute masking model
        ml_output = self.mlmodel(**inputs)
        ml_loss, ml_logits = ml_output

        masked_word_idx = (inputs['labels'] != -100)[0]
        cl_logits = self.classifier(ml_logits[:, masked_word_idx, :].view(-1, len(self.tokenizer)))

        self.loss(cl_logits, )

        

        top2_idx = torch.topk(ml_logits, k=2)
        top2_predictions = 

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
