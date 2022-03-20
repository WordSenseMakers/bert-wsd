from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset

import pandas as pd


class MaskedLMWithSynsetClassification(nn.Module):
    def __init__(
        self,
        mlmodel: PreTrainedModel,
        mltokenizer: PreTrainedTokenizer,
        tcmodel: PreTrainedModel,
        tctokenizer: PreTrainedTokenizer,
        label_count: int,
    ):
        super(MaskedLMWithSynsetClassification, self).__init__()
        self.mlmodel = mlmodel
        self.mltokenizer = mltokenizer
        self.tcmodel = tcmodel
        self.tctokenizer = tctokenizer
        self.dropout = nn.Dropout(0.1)
        self.label_count = label_count

        self.mlunmasker = pipeline(
            task="fill-mask", model=self.model, tokenizer=self.tokenizer
        )

    def forward(self, **inputs):
        sentence_idx = inputs.pop("sentence_idx")
        sentence_idx_df = pd.DataFrame({"sentence_idx": sentence_idx})
        lossable = pd.merge(
            self.semcor_dataset.masked, sentence_idx_df, how="inner", on="sentence_idx"
        )

        # Execute masking model
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

        mlpredictions = with_gold_standard.token_str

        tc_ds = Dataset.from_pandas(mlpredictions).map(
            lambda chunk: self.tctokenizer(
                chunk.token_str, padding="longest", truncation="longest_first"
            ),
            batched=True,
        )
        tc_output = self.tcmodel(tc_ds)

        loss, logits = tc_output
        assert "labels" in inputs
        cel = nn.CrossEntropyLoss()(logits.view(-1, self.label_count), labels.view(-1))

        # TODO: Can we do this without the hidden_states and attentions arguments?
        # TODO: Is it OK to return TokenClassifierOutput even though we want to train a Masked Language Model?
        return TokenClassifierOutput(loss=cel, logits=logits)
