import fsspec.asyn
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
class CLEFModel(nn.Module):
    def __init__(self, factual_branch, num_classes=7):
        """
        Initializes the CLEF model.

        Args:
            factual_branch: Any CAER model for factual predictions Y_e(X).
            num_classes: Number of emotion classes for prediction.
        """
        super(CLEFModel, self).__init__()

        # Factual branch (any CAER model)
        self.factual_branch = factual_branch

        # Context branch (ResNet-50 with final FC replaced by Identity)
        self.context_branch = models.resnet152(pretrained=True)
        num_features = self.context_branch.fc.in_features  # Get the number of input features to the fc layer
        self.context_branch.fc = nn.Linear(num_features, num_classes)
        # Number of classes

        # Counterfactual branch (trainable parameter)
        self.counterfactual_feature = nn.Parameter(torch.randn(7))

    def forward(self, full_image, face_img, context_img):

        y_factual = self.factual_branch(face_img, full_image)
        y_factual_softmax = F.softmax(y_factual, dim=1)

        y_context = self.context_branch(context_img)
        y_context_softmax = F.softmax(y_context, dim=1)

        #TODO[ybkaratas]: why does this need face_img? Check...
        y_counterfactual = self.counterfactual_feature.unsqueeze(0).expand(face_img.size(0), -1)
        y_counterfactual_softmax = F.softmax(y_counterfactual, dim=1)

        factual_fused = F.softmax(self.fusion(y_factual_softmax, y_context_softmax), dim=1)
        counterfactual_fused = F.softmax(self.fusion(y_counterfactual_softmax, y_context_softmax), dim=1)

        """ print("FACT: ", factual_fused)
        print("COUN: ", counterfactual_fused)"""
        output = F.softmax((factual_fused - counterfactual_fused), dim=1)

        return output, factual_fused, counterfactual_fused



    def fusion(self, y_c, y_e):

        fused = torch.log(torch.sigmoid(y_c + y_e))  # Apply log-sigmoid
        return fused
