import random
from torchvision.datasets import CIFAR10

class NoisyCIFAR10(CIFAR10):
    def __init__(self, *args, noise_ratio=0.1, num_classes=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_ratio = noise_ratio
        self.num_classes = num_classes
        self.apply_label_noise()

    def apply_label_noise(self):
        num_samples = len(self.targets)
        num_noisy = int(num_samples * self.noise_ratio)

        noisy_indices = random.sample(range(num_samples), num_noisy)
        for idx in noisy_indices:
            original_label = self.targets[idx]
            noisy_label = random.choice([x for x in range(self.num_classes)])
            self.targets[idx] = noisy_label

        print(f"Applied noise to {num_noisy} out of {num_samples} samples.")
