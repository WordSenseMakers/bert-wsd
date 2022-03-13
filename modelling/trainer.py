from transformers import Trainer, pipeline
import numpy as np, pandas as pd

from datagen.dataset import SemCorDataSet

from . import metrics


class WordSenseTrainer(Trainer):
    def __init__(
        self,
        semcor_dataset: SemCorDataSet,
        sentence_level: pd.DataFrame,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.semcor_dataset = semcor_dataset
        self.sentence_level = sentence_level
        self.word_sense = metrics.WordSenseSimilarity(
            config_name="max", dataset=self.semcor_dataset
        )

        # self.unmasker = pipeline(task="fill-mask", model=self.model, tokenizer=self.tokenizer)

    def compute_loss(self, model, inputs: dict, return_outputs=False):
        ids, attention_mask, ttis = (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
        )
        model_outputs = model(
            input_ids=ids, attention_mask=attention_mask, token_type_ids=ttis
        )
        # _, logits = model_outputs
        print(inputs, model_outputs)

        metric_output = self.word_sense.compute(
            predictions=model_outputs, references=None
        )

        loss = metric_output["loss"]

        return (loss, model_outputs) if return_outputs else loss
