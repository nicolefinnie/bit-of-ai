from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar_dataloader(img_size: int = 32, batch_size: int = 32):
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    calibration_dataset = datasets.CIFAR10(root='.data', train=True, download=True, transform=transform)
    calibration_data_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=True)
    return calibration_data_loader