import string

from transformers import Trainer, PreTrainedTokenizer, pipeline
import numpy as np, pandas as pd

import datasets, nltk
from nltk.corpus import wordnet as wn

from datagen.dataset import SemCorDataSet
from . import metrics


class WordSenseTrainer(Trainer):
    def __init__(
        self,
        dataset: SemCorDataSet,
        # sentence_level: pd.DataFrame,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.semcor_dataset = dataset
        # self.sentence_level = sentence_level
        """ self.word_sense = metrics.WordSenseSimilarity(
            model=self.model, tokenizer=self.tokenizer, dataset=self.semcor_dataset
        ) """

        self.unmasker = pipeline(
            task="fill-mask", model=self.model, tokenizer=self.tokenizer
        )
        nltk.download("omw-1.4")

        # self.unmasker = pipeline(task="fill-mask", model=self.model, tokenizer=self.tokenizer)

    def compute_loss(self, model, inputs: dict, return_outputs=False):
        sentence_idx = inputs.pop("sentence_idx")
        model_outputs = model(**inputs)
        labels = inputs.get("labels")

        sentence_idx_ds = pd.DataFrame({"sentence_idx": sentence_idx})
        lossable = pd.merge(
            self.semcor_dataset.masked, sentence_idx_ds, how="inner", on="sentence_idx"
        )

        # Mask tokens available in dataset,
        # creating one sentence per masked token with said token masked
        unmasked = self.unmasker(lossable["masked"].tolist(), top_k=2)
        # predictions = [(prediction[0]["token_str"] if prediction[0]["token_str"] != token]

        scores = []

        for (pred1, pred2), reference, sense_keys, sentence in zip(
            unmasked, lossable["token"], lossable["sense-keys"], lossable["masked"]
        ):
            p1t, p2t = pred1["token_str"].strip(), pred2["token_str"].strip()

            # Avoid overfitting and punctuation predictions
            prediction = (
                p1t if p1t != reference and p1t not in string.punctuation else p2t
            )

            print(prediction, reference)

            # Load synset
            pred_synsets = wn.synsets(prediction)
            if not pred_synsets:
                # Determine score based on word embeddings
                raise Exception(
                    f"Guessed {prediction} for {sentence}, but needs word embedding lookup"
                )

            else:
                # Determine score based on word sense
                sense_keys = sense_keys.split(",")
                actual_synsets = (
                    wn.lemma_from_key(sense_key).synset() for sense_key in sense_keys
                )
                # TODO: Confer on how to select which similarity
                score = max(
                    wn.path_similarity(pred, actual_synset)
                    for pred in pred_synsets
                    for actual_synset in actual_synsets
                )
                # Higher similarity indicates a lesser need for adjustment, so invert (?)
                scores.append(1 - score)

        loss = sum(scores)
        return (loss, model_outputs) if return_outputs else loss
