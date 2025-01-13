import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from oauthlib.oauth2.rfc6749.parameters import parse_implicit_response

from .resnet152_places365 import resnet152_places365
from .resnet152_places365 import Lambda





class CLEFModelMultiLabel(nn.Module):
    def __init__(self, factual_branch, num_classes=26, resnet_model_path = None):
        super(CLEFModelMultiLabel, self).__init__()
        self.factual_branch = factual_branch
        self.num_classes = num_classes

        self.context_branch = resnet152_places365
        self.context_branch.load_state_dict(torch.load(resnet_model_path))

        self.context_branch = self.modify_model_output(self.context_branch, num_classes)

        # Counterfactual branch
        self.counterfactual_feature = nn.Parameter(torch.randn(num_classes))

        # Projection layer for factual branch (if output dimensions don't match)
        #self.factual_projection = nn.Linear(26, num_classes)


    def forward(self, full_image, face_img, context_img):
        # Factual branch output
        y_factual = self.factual_branch(face_img, full_image)
        y_factual_sigmoid = torch.sigmoid(y_factual)


        # Context branch output
        y_context = self.context_branch(context_img)
        y_context_sigmoid = torch.sigmoid(y_context)

        # Counterfactual branch
        y_counterfactual = self.counterfactual_feature.unsqueeze(0).expand(face_img.size(0), -1)
        y_counterfactual_sigmoid = torch.sigmoid(y_counterfactual)

        # Fusion
        factual_fused =  torch.sigmoid(self.fusion(y_factual_sigmoid, y_context_sigmoid))
        counterfactual_fused = torch.sigmoid(self.fusion(y_counterfactual_sigmoid, y_context_sigmoid))

        # Final output
        output = torch.sigmoid(factual_fused - counterfactual_fused)

        return output, factual_fused, counterfactual_fused

    def fusion(self, y_c, y_e):
        fused = torch.log(y_c + y_e)  # Apply log-sigmoid
        return fused

    def modify_model_output(self, model, num_outputs):
        """
        Modify the final layer of the model to change the number of outputs.

        Args:
            model (nn.Sequential): The ResNet model.
            num_outputs (int): The desired number of outputs.

        Returns:
            nn.Sequential: The modified model.
        """
        # Replace the last linear layer
        layers = list(model.children())
        final_layer = layers[-1]

        if isinstance(final_layer, nn.Sequential):
            inner_layers = list(final_layer.children())
            if isinstance(inner_layers[-1], nn.Linear):
                # Replace the final linear layer
                in_features = inner_layers[-1].in_features
                inner_layers[-1] = nn.Linear(in_features, num_outputs)
                layers[-1] = nn.Sequential(*inner_layers)
            else:
                raise ValueError("Expected the last layer to be nn.Linear")
        else:
            raise ValueError("Expected the last layer to be nn.Sequential")

        return nn.Sequential(*layers)



# Modify the model

class CLEFModel(nn.Module):
    def __init__(self, factual_branch, resnet_model_path, num_classes=7):
        """
        Initializes the CLEF model.

        Args:
            factual_branch: Any CAER model for factual predictions Y_e(X).
            num_classes: Number of emotion classes for prediction.
        """
        super(CLEFModel, self).__init__()

        # Factual branch (any CAER model)
        self.factual_branch = factual_branch

        self.context_branch = resnet152_places365
        self.context_branch.load_state_dict(torch.load(resnet_model_path))

        #num_features = self.context_branch.fc.in_features  # Get the number of input features to the fc layer
        # self.context_branch.fc = nn.Linear(num_features, num_classes)

        self.context_branch = self.modify_model_output(self.context_branch, num_classes)

        # Number of classes

        # Counterfactual branch (trainable parameter)
        self.counterfactual_feature = nn.Parameter(torch.randn(7))

    def forward(self, full_image, face_img, context_img):

        y_factual = self.factual_branch(face_img, full_image)
        y_factual_softmax = F.softmax(y_factual, dim=1)

        y_context = self.context_branch(context_img)
        y_context_softmax = F.softmax(y_context, dim=1)

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

    def modify_model_output(self, model, num_outputs):

        layers = list(model.children())
        final_layer = layers[-1]

        if isinstance(final_layer, nn.Sequential):
            inner_layers = list(final_layer.children())
            if isinstance(inner_layers[-1], nn.Linear):
                # Replace the final linear layer
                in_features = inner_layers[-1].in_features
                inner_layers[-1] = nn.Linear(in_features, num_outputs)
                layers[-1] = nn.Sequential(*inner_layers)
            else:
                raise ValueError("Expected the last layer to be nn.Linear")
        else:
            raise ValueError("Expected the last layer to be nn.Sequential")

        return nn.Sequential(*layers)
