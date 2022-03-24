import abc, statistics
from typing import Any, Dict, Optional, Union

import datasets
from nltk.corpus import wordnet as wn
import numpy as np

from datagen.dataset import SemCorDataSet


class WordSenseSimilarity(datasets.Metric):
    def __init__(
        self,
        dataset: SemCorDataSet,
        config_name: Optional[str] = None,
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
        super().__init__(
            config_name,
            keep_in_memory,
            cache_dir,
            num_process,
            process_id,
            seed,
            experiment_id,
            max_concurrent_cache_files,
            timeout,
            **kwargs,
        )
        self.dataset = dataset

    def _info(self):
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description="Compute different sense similarities for references (gold-standard) and predicted words",
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

        synset_from_sense_key = np.vectorize(lambda x: wn.lemma_from_key(x).synset())
        path_sim = np.vectorize(wn.path_similarity)

        pred_synset = synset_from_sense_key(predictions)
        ref_synsets = synset_from_sense_key(references)
        similarities = path_sim(pred_synset, ref_synsets)

        return {"word_sense_distance": similarities.sum()}
