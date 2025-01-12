from torch.utils.data import random_split

def create_train_val_split(dataset, val_ratio=0.1):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    return random_split(dataset, [train_size, val_size])
