from torchvision import transforms
import torch
from PIL import Image

def get_preprocessing_transforms(image_size=224):
    """
    Returns the preprocessing transforms for images.

    Args:
        image_size (int): The target size to resize the images (default is 224x224).

    Returns:
        torchvision.transforms.Compose: Composed preprocessing transforms.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize images to a fixed size
        transforms.ToTensor(),                       # Convert PIL Image to Tensor
        transforms.Normalize(                        # Normalize to match ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(image, transforms_pipeline):
    """
    Preprocess a single image using the provided transforms.

    Args:
        image (PIL.Image or str): The input image or the path to the image.
        transforms_pipeline (torchvision.transforms.Compose): The preprocessing pipeline.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    return transforms_pipeline(image)


def preprocess_batch(images, transforms_pipeline):
    """
    Preprocess a batch of images.

    Args:
        images (list of PIL.Image or str): List of images or paths to the images.
        transforms_pipeline (torchvision.transforms.Compose): The preprocessing pipeline.

    Returns:
        torch.Tensor: Batch of preprocessed image tensors.
    """
    return torch.stack([preprocess_image(img, transforms_pipeline) for img in images])


# Example usage
if __name__ == "__main__":
    # Example images
    from io import BytesIO
    import requests

    url = "https://upload.wikimedia.org/wikipedia/commons/4/41/Siberian_Husky_pho.jpg"
    response = requests.get(url)
    example_image = Image.open(BytesIO(response.content))

    # Get preprocessing transforms
    transforms_pipeline = get_preprocessing_transforms()

    # Preprocess a single image
    preprocessed_image = preprocess_image(example_image, transforms_pipeline)
    print(f"Preprocessed single image shape: {preprocessed_image.shape}")

    # Preprocess a batch of images
    batch = [example_image, example_image]  # Duplicate for batch example
    preprocessed_batch = preprocess_batch(batch, transforms_pipeline)
    print(f"Preprocessed batch shape: {preprocessed_batch.shape}")
