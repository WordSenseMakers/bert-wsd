from transformers import Trainer, pipeline
import numpy as np, pandas as pd

from datagen.dataset import SemCorDataSet

from . import metrics


class WordSenseTrainer(Trainer):
    def __init__(
        self,
        dataset: SemCorDataSet,
        # sentence_level: pd.DataFrame,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.semcor_dataset = dataset
        # self.sentence_level = sentence_level
        self.word_sense = metrics.WordSenseSimilarity(
            config_name="max", dataset=self.semcor_dataset
        )

        # self.unmasker = pipeline(task="fill-mask", model=self.model, tokenizer=self.tokenizer)

    def compute_loss(self, model, inputs: dict, return_outputs=False):
        model_outputs = model(**inputs)
        sentences = self.tokenizer.batch_decode(inputs["input_ids"])
        # print(sentences)

        logits = model_outputs.get("logits").detach().numpy()
        labels = inputs.get("labels")
        predictions = np.argmax(logits, axis=-1)

        metric_output = self.word_sense.compute(
            predictions=predictions, references=labels
        )

        loss = metric_output["word_sense_distance"]
        return (loss, model_outputs) if return_outputs else loss
