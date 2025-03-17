import torch
import torch.nn as nn
from transformers import ViTForImageClassification


class FireSmokeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(FireSmokeDetector, self).__init__()
        # Load the pre-trained ViT model with `ignore_mismatched_sizes=True`
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # Ignore size mismatch in the classifier layer
        )

    def forward(self, x):
        return self.vit(x)


# Example usage
if __name__ == "__main__":
    model = FireSmokeDetector(num_classes=2)
    print(model)
