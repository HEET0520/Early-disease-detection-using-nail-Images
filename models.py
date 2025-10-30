import torch
import torch.nn as nn
import torchvision.models as models

class WeightedAvgFeatureExtractor(nn.Module):
    """
    Feature extractor based on Cell 24.
    Concatenates features from MobileNetV2, EfficientNet-B0, 
    and SqueezeNet-1.1, then projects them to 512 dimensions.
    """
    def __init__(self):
        super(WeightedAvgFeatureExtractor, self).__init__()

        # 1. Load pretrained models
        self.base_model_1 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.base_model_2 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.base_model_3 = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)

        # 2. Remove the final classifier layers
        self.base_model_1.classifier = nn.Identity()
        self.base_model_2.classifier = nn.Identity()
        self.base_model_3.classifier = nn.Identity()

        # 3. Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 4. Projection layer
        # MobileNetV2 outputs 1280 features
        # EfficientNet-B0 outputs 1280 features
        # SqueezeNet outputs 512 features
        combined_feature_size = 1280 + 1280 + 512
        
        # Project combined features to 512 (to match PCA input from cell 16)
        self.projection = nn.Linear(combined_feature_size, 512) 

    def forward(self, x):
        # Get features
        feat1 = self.global_pool(self.base_model_1.features(x)).view(x.size(0), -1)
        feat2 = self.global_pool(self.base_model_2.features(x)).view(x.size(0), -1)
        feat3 = self.global_pool(self.base_model_3.features(x)).view(x.size(0), -1)

        # Concatenate features
        combined_feat = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Project to final 512-dimension feature vector
        feat = self.projection(combined_feat)
        return feat