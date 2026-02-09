# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from typing import Any, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

try:
    from torchdata.stateful_dataloader import StatefulDataLoader
except ImportError as e:
    raise ImportError(
        "Please install the latest torchdata nightly to use StatefulDataloader via:"
        "pip3 install --pre torchdata --index-url https://download.pytorch.org/whl/nightly"
    ) from e

from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging_utils import logger

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

# map from dataset name to a local directory, or
# a dataset repository on the HF hub
_supported_datasets = {
    "c4_mini": "torchtitan/datasets/c4_mini",
    "c4": "allenai/c4",
    "tulu_sft": "allenai/tulu-3-sft-mixture",
}


class HuggingFaceDataset(IterableDataset, Stateful):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]):
            Path to the dataset in the file system. If provided, data will be loaded
            from this path instead of downloaded.
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    We currently support the c4 dataset and a subset of it:
    c4_mini (45K training entries)
    c4 (177M training entries - this dataset is streamed due to the size)

    >> c4 (EN) <<:
    c4 cleaned, English version
    Data input format (c4):
    {
    'url': 'https://klyq.com/beginners-bbq-class-taking-place-in-missoula/',
    'text': 'Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at ...',
    'timestamp': '2019-04-25T12:57:54Z'
    }

    Example use (c4):
    >>> ds = HuggingFaceDataset(dataset_name="c4", dataset_path=None, tokenizer=tokenizer)
    >>> for batch in Dataloader(ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        # allow user to pass in a (local or HF hub) path to use unsupported datasets
        if dataset_name not in _supported_datasets:
            if dataset_path:
                logger.warning(
                    f"Dataset {dataset_name} is not tested or verfied. "
                    f"Recommended datasets are: {list(_supported_datasets.keys())}."
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} is not supported. "
                    f"Supported datasets are: {list(_supported_datasets.keys())}."
                )

        if not dataset_path:
            dataset_path = _supported_datasets[dataset_name]
        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")

        if dataset_name == "c4":
            # c4 is huge, and requires both streaming and language selection
            # (we default to en)
            ds = load_dataset(dataset_path, name="en", split="train", streaming=True)
        elif dataset_name == "c4_mini":
            ds = load_dataset(dataset_path, split="train")
        elif dataset_name == "tulu_sft":
            # SFT dataset - load with streaming for large datasets
            ds = load_dataset(dataset_path, split="train", streaming=True)
        elif dataset_name == "mathinstruct":
            # MathInstruct dataset - load from local JSONL file
            # Since it's a local file, we need to load it as a JSONL dataset
            # Use streaming=False to enable epoch-based training
            ds = load_dataset("json", data_files=dataset_path, split="train", streaming=False)
        else:
            ds = load_dataset(dataset_path, split="train", streaming = True)

        # TODO: support shuffling and checkpointing
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite

        # variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []
        self._all_masks: List[int] = []  # Track mask for each token
        
        # Store original dataset for size calculation
        self._original_dataset = ds

    def get_dataset_size(self) -> int:
        """Get the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset, or -1 if size cannot be determined
        """
        return len(self._original_dataset)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                if self.dataset_name == "tulu_sft":
                    # Handle SFT dataset format
                    sample_text = self._process_sft_sample(sample)
                    mask_info = None
                elif self.dataset_name == "mathinstruct":
                    # Handle MathInstruct dataset format
                    sample_text, mask_info = self._process_mathinstruct_sample(sample)
                else:
                    # Handle regular text datasets (c4, etc.)
                    sample_text = sample["text"]
                    mask_info = None
                
                # Tokenize the sample
                sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                
                # Create mask for this sample's tokens
                if mask_info and mask_info.get("mask_instruction", False):
                    # Create mask for this sample
                    instruction_token_count = mask_info.get("instruction_token_count", 0)
                    # Mask instruction tokens (including BOS token)
                    mask_length = min(instruction_token_count + 1, len(sample_tokens))  # +1 for BOS
                    sample_masks = [0] * mask_length + [1] * (len(sample_tokens) - mask_length)
                else:
                    sample_masks = [1] * len(sample_tokens)  # All unmasked for non-masked datasets
                
                self._all_tokens.extend(sample_tokens)
                self._all_masks.extend(sample_masks)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    mask = torch.LongTensor(self._all_masks[:max_buffer_token_len])
                    
                    # update tokens and masks to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    self._all_masks = self._all_masks[max_buffer_token_len:]
                    
                    input = x[:-1]
                    label = x[1:]
                    label_mask = mask[1:]  # Shift mask to match label
                    
                    # Always return 3 elements for consistency in batching
                    # If no masking is needed, label_mask will be all 1s
                    yield input, label, label_mask

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data.")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(
                    f"Dataset {self.dataset_name} is being re-looped. "
                    "Loss related metrics might be misleading."
                )

    def _process_sft_sample(self, sample):
        """Process SFT dataset sample into training text format.
        
        SFT datasets typically contain conversation format data.
        This method converts the conversation into a single text string
        suitable for language modeling training.
        """
        # Handle different possible SFT dataset formats
        if "messages" in sample:
            # Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
            messages = sample["messages"]
            text_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    text_parts.append(f"Human: {content}")
                elif role == "assistant":
                    text_parts.append(f"Assistant: {content}")
                elif role == "system":
                    text_parts.append(f"System: {content}")
            return "\n".join(text_parts)
        elif "conversations" in sample:
            # Alternative format with conversations key
            conversations = sample["conversations"]
            text_parts = []
            for conv in conversations:
                role = conv.get("from", conv.get("role", ""))
                content = conv.get("value", conv.get("content", ""))
                if role == "human":
                    text_parts.append(f"Human: {content}")
                elif role == "gpt":
                    text_parts.append(f"Assistant: {content}")
            return "\n".join(text_parts)
        elif "text" in sample:
            # Fallback to text field if available
            return sample["text"]
        else:
            # If no recognized format, convert the entire sample to string
            logger.warning(f"Unknown SFT sample format: {list(sample.keys())}")
            return str(sample)

    def _process_mathinstruct_sample(self, sample):
        """Process MathInstruct dataset sample into training text format.
        
        MathInstruct datasets contain instruction-output pairs for math problems.
        This method formats them for language modeling training with instruction masking.
        
        Expected format:
        {
            "source": "data/CoT/aqua_rat.json",
            "instruction": "The distance between two stars is 6.52 × 10^5 light years...",
            "output": "Let's think about the multi-choice question..."
        }
        
        Returns:
            tuple: (formatted_text, mask_info) where mask_info indicates which tokens to mask
        """
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        
        # Format as instruction-following conversation
        formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        # Calculate precise token-level mask information
        # We need to tokenize the instruction part with the same settings as the full text
        instruction_part = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize instruction part with BOS=True to match the full text tokenization
        instruction_tokens = self._tokenizer.encode(instruction_part, bos=True, eos=False)
        instruction_token_count = len(instruction_tokens)
        
        mask_info = {
            "mask_instruction": True,
            "instruction_token_count": instruction_token_count,
            "instruction_text": instruction_part
        }
        
        return formatted_text, mask_info

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        # Skip samples
        if isinstance(self._data, IterableDataset):
            it = iter(self._data)
            # Naively iterate through the samples as skip may not be supported
            for _ in range(self._sample_idx):
                next(it)
            return it

        # As skipping to the end throws an error in case of map-style dataset, return an empty iterator
        if self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int):
        super().__init__(hf_ds, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid, don't log a warning
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}."
            )
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    infinite: bool = True,
):
    hf_ds = HuggingFaceDataset(
        dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, infinite
    )

    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)


