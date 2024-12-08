import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

def pairs_to_num(xi, n):

    """
        Convert one long input with n-features encoding to n^2 pairs encoding.
    """
    ln = len(xi)
    xin = torch.zeros(ln // 2)
    for ii, xii in enumerate(xi):
        xin[ii // 2] += xii * n ** (1 - ii % 2)
    return xin


def pairing_features(x, n):
    """
        Batch of inputs from n to n^2 encoding.
    """
    xn = torch.zeros(x.shape[0], x.shape[-1] // 2)
    for i, xi in enumerate(x.squeeze()):
        xn[i] = pairs_to_num(xi, n)
    return xn

def number2base(numbers, base, string_length=None):
    digits = []
    while numbers.sum():
        digits.append(numbers % base)
        numbers = numbers.div(base, rounding_mode='floor')
    if string_length:
        assert len(digits) <= string_length, "String length required is too small to represent numbers!"
        digits += [torch.zeros(len(numbers), dtype=int)] * (string_length - len(digits))
    return torch.stack(digits[::-1]).t()


def dec2bin(x, bits=None):
    """
    Convert decimal number to binary.
    :param x: decimal number
    :param bits: number of bits to use. If `None`, the minimum possible is used.
    :return: x in binary
    """
    if bits is None:
        bits = (x.max() + 1).log2().ceil().item()
    x = x.int()
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def unique(x, dim=-1):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)

def cifar_10_download(path="./data", transform=None):
    """Download CIFAR10 dataset using torchvision downloaders

    Args:
        path (str or os.path , optional): Overrides torchvision default dataloader location. Defaults to ./data.
        transform (torchvision.transforms.Compose, optional): Override default ToTensor transformation. Can be used for data preprocessing. Defaults to None.

    Returns:
        train_dataset, test_dataset (CIFAR10, CIFAR10) : CIFAR10 datasets for train and test.
    """
    if not transform:
        transform = transforms.Compose(transforms=[transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=path,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


def cifar_10_load(dataset, batch_size=64, shuffle=True, num_workers=1):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader


if __name__ == "__main__":
    train_dataset, test_dataset = cifar_10_download()

    train_loader = cifar_10_load(train_dataset)
    test_loader = cifar_10_load(test_dataset, shuffle=False)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    for images, labels in train_loader:
        print(f"Batch size: {images.size()}, Labels: {labels.size()}")
        break

