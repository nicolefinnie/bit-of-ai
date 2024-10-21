from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


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
    dataset = datasets.CIFAR10(root='.data', train=True, download=True, transform=transform)
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