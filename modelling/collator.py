from typing import Union, List, Any, Dict

import torch
from transformers import DataCollatorForWholeWordMask


class BetterDataCollatorForWholeWordMask(DataCollatorForWholeWordMask):

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super(BetterDataCollatorForWholeWordMask, self).torch_call(examples)
        for input_key in examples[0].keys():
            if input_key not in batch:
                all_of_batch = [e[input_key] for e in examples]
                batch[input_key] = torch.stack(all_of_batch, dim=0)
        return batch
