import abc, statistics
from typing import Any, Dict, Optional, Union

import datasets
from nltk.corpus import wordnet as wn

from datagen.dataset import SemCorDataSet


class WordSenseSimilarity(datasets.Metric):
    def __init__(
        self,
        dataset: SemCorDataSet,
        # max, avg or min
        config_name: str,
        keep_in_memory: bool = False,
        cache_dir: Optional[str] = None,
        num_process: int = 1,
        process_id: int = 0,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        max_concurrent_cache_files: int = 10000,
        timeout: Union[int, float] = 100,
        **kwargs,
    ):
        super(datasets.Metric, self).__init__(
            config_name,
            keep_in_memory,
            cache_dir,
            num_process,
            process_id,
            seed,
            experiment_id,
            max_concurrent_cache_files,
            timeout,
        )
        self.dataset = dataset

    def _info(self):
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description="Compute different sense similarities for references (gold-standard) and predicted words. Will fallback to cosine similarity over word embeddings",
            citation=None,
            inputs_description="predictions as word, gold-standard by lemma-key, "
            + "stored in an iterable ordered structure",
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            # Homepage of the metric for documentation
            homepage=None,
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/WordSenseMakers/bert-wsd"],
            reference_urls=None,
        )

    def _compute(
        self, *, predictions=None, references=None, **kwargs
    ) -> Dict[str, Any]:
        print(predictions)
        print(references)
        # Compute possible synsets for a given prediction
        pred_synsets = [wn.synsets(prediction) for prediction in predictions]
        ref_synsets = [
            wn.lemma_from_key(sense_key).synset() for sense_key in references
        ]

        if self.config_name == "max":
            selector = max
        elif self.config_name == "min":
            selector = min
        elif self.config_name == "avg":
            selector = statistics.fmean

        similarities = [
            selector(wn.path_similarity(pred, ref_synset) for pred in pred_synset_set)
            for pred_synset_set, ref_synset in zip(pred_synsets, ref_synsets)
        ]

        return {self.config_name: statistics.fmean(similarities)}
