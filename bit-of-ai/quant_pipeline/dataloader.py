import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import GPT2TokenizerFast, AutoTokenizer


def get_cifar_dataloader(img_size: int = 32, batch_size: int = 32, split='calibration') -> DataLoader:
    """Get cifar dataloader for calibration or validation.

    Args:
        img_size (int, optional): Defaults to 32.
        batch_size (int, optional): Defaults to 32.
        split (str, optional): get calibration or validation data spilt. Defaults to 'calibration'.

    Raises:
        ValueError: non valid split

    Returns:
        DataLoader: dataloader
    """

    if isinstance(img_size, int):
        resized_transform = transforms.Resize((img_size, img_size))
    else:
        resized_transform = transforms.Resize(img_size)

    transform = transforms.Compose([resized_transform, transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    calibration_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - calibration_size
    calibration_dataset, valiadtion_dataset = random_split(dataset, [calibration_size, validation_size])
    if split == 'calibration':
        dataloader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=True)
    elif split == 'validation':
        dataloader = DataLoader(valiadtion_dataset, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError(f"Invalid split {split}. Must be either 'calibration' or 'validation'")
    return dataloader


class WikiTextDataset(Dataset):
    """Torch Dataset built upon wikitext."""

    def __init__(self, hf_dataset: DatasetDict, tokenizer: GPT2TokenizerFast, max_length: int = 512) -> None:
        """Build wikitext dataset.
        Args:
            hf_dataset (DatasetDict): huggingface dataset
            tokenizer (GPT2TokenizerFast): tokenizer
            max_length (int, optional): max length of each sequence for training. Defaults to 512.
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Get the number of data samples.
        Returns:
            int: The number of data samples.
        """
        return len(self.dataset)

    # calculate the max length of the dataset
    def shift_tokens_right(self, input_ids: list[int], pad_token_id: int) -> list[int]:
        """Shift tokens to the right by one for label data.
        Args:
            input_ids (list[int]): token ID list
            pad_token_id (int): _padded token
        Returns:
            list[int]: padded token ID list
        """
        return [pad_token_id] + input_ids[:-1]

    def __getitem__(self, idx: int) -> DatasetDict:
        """Get the item at the idx index.
        Args:
            idx (int): index of the sequence
        Returns:
            DatasetDict: a single dataset
        """
        text = self.dataset[idx]["text"]

        # Tokenize and truncate/pad
        tokenized_inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            # returns tensors with an additional batch dimension, (1, seq_len)
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = tokenized_inputs["input_ids"].squeeze()  # squeeze to (seq_len)
        attention_mask = tokenized_inputs["attention_mask"].squeeze()  # squeeze to (seq_len) as well
        labels = self.shift_tokens_right(input_ids.tolist(), self.tokenizer.pad_token_id)
        labels = torch.tensor(labels)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def get_wikitext_dataloader(
    dataset: DatasetDict = None,
    batch_size: int = 1,
    split: str = "cailbration",
    tokenizer: GPT2TokenizerFast = None,
    max_seq_length: int = 1024,
    is_train: bool = False,
    num_samples: int = -1,
) -> DataLoader:
    """Create pytorch distributed dataloader.
    Args:
        dataset (Dataset): dataset from hugging face
        batch_size (int): batch size
        split (str): calibration/validation
        tokenizer (GPT2TokenizerFast): tokenizer, tiktoken GPT2 by default
        max_seq_length (int, optional): max training sequence length. Defaults to 1024.
        is_train (bool): shuffle or not
        num_samples (int): How many samples we should load, if negative,
            we load all samples.
    Returns:
        DataLoader: dataloader
    """
    split = "train" if split == "calibration" else split
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # GPT2 wasn't trained with a dedicated padding token, so we can use whatever we want.
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-v1") if dataset is None else dataset
    dataset = dataset[split].select(range(int(num_samples))) if num_samples > 0 else dataset[split]
    dataset = WikiTextDataset(dataset, tokenizer, max_seq_length)
    sampler = Sampler(dataset, shuffle=is_train)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=True,
    )
    return dataloader