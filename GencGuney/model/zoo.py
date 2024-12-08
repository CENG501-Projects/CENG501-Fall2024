import torch
import torchvision.models as models
from torchvision import transforms

class ModelZoo:
    """
    A class to manage a collection of pre-trained models (model zoo) for OOD detection.

    Attributes:
        models (list): List of pre-trained models.
        preprocess (callable): A preprocessing function for input data.
        device (torch.device): Device to run models (CPU/GPU).
    """

    def __init__(self, device=None):
        self.models = []
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to the size expected by pre-trained models
            transforms.ToTensor(),         # Convert to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_model(self, model, name):
        """
        Add a pre-trained model to the zoo.

        Args:
            model (torch.nn.Module): The pre-trained model.
            name (str): The name of the model.
        """
        model.to(self.device).eval()
        self.models.append({"model": model, "name": name})

    def initialize_zoo(self):
        """
        Initialize the model zoo with a variety of pre-trained models.
        """
        print("Initializing Model Zoo...")
        # Add pre-trained models from torchvision
        self.add_model(models.resnet18(pretrained=True), "ResNet18")
        self.add_model(models.resnet34(pretrained=True), "ResNet34")
        self.add_model(models.resnet50(pretrained=True), "ResNet50")
        self.add_model(models.densenet121(pretrained=True), "DenseNet121")
        self.add_model(models.resnet18(pretrained=True), "ResNet18_Contrastive")  # Placeholder for contrastive loss
        print("Model Zoo initialized with the following models:")
        for model in self.models:
            print(f"- {model['name']}")

    def get_model_names(self):
        """
        Get the names of the models in the zoo.

        Returns:
            list: Names of the models.
        """
        return [model["name"] for model in self.models]

    def inference(self, inputs):
        """
        Run inference on inputs using all models in the zoo.

        Args:
            inputs (torch.Tensor): Batch of input images.

        Returns:
            dict: Dictionary of results with model names as keys and logits as values.
        """
        inputs = inputs.to(self.device)
        results = {}
        for model_info in self.models:
            model = model_info["model"]
            name = model_info["name"]
            with torch.no_grad():
                logits = model(inputs)
            results[name] = logits.cpu()
        return results

    def preprocess_inputs(self, raw_images):
        """
        Preprocess a batch of raw images for model inference.

        Args:
            raw_images (list of PIL.Image): List of raw images.

        Returns:
            torch.Tensor: Preprocessed images as a batch tensor.
        """
        return torch.stack([self.preprocess(image) for image in raw_images])


# Example usage
if __name__ == "__main__":
    from PIL import Image
    import requests
    from io import BytesIO

    # Load an example image
    url = "https://upload.wikimedia.org/wikipedia/commons/4/41/Siberian_Husky_pho.jpg"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    # Initialize Model Zoo
    model_zoo = ModelZoo()
    model_zoo.initialize_zoo()

    # Preprocess image
    inputs = model_zoo.preprocess_inputs([image])

    # Run inference
    results = model_zoo.inference(inputs)
    print("Inference Results:")
    for model_name, logits in results.items():
        print(f"{model_name}: {logits[:5]}")  # Display first 5 logits for each model
