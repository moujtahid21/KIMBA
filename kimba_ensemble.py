import torch
import torch.nn as nn
import os

class KIMBAEnsemble(nn.Module):
    """
    Ensemble model for tumor detection.
    Combines the predictions of several PyTorch models.
    """

    def __init__(self):
        super(KIMBAEnsemble, self).__init__()
        self.models = nn.ModuleList()
        self.class_names = None  # To be defined upon loading

    def load_saved_models(self, models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        model_files.sort()
        if len(model_files) < 2:
            raise ValueError("At least two models must be saved in the folder.")

        self.models = nn.ModuleList()
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            # Adapt here according to the model name
            if "vgg" in model_file.lower():
                from torchvision.models import vgg19
                model = vgg19(weights=None)
                # Adapt the number of classes if needed
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, 3)  # 3 classes example
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
            elif "resnet" in model_file.lower():
                from torchvision.models import resnet50
                model = resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes exemple
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
            else:
                raise ValueError(f"Unknown model for file {model_file}")
            model.eval()
            self.models.append(model)
        # Optional: define class names
        self.class_names = ["benign", "malignant", "normal"]  # Adapt according to your dataset

    def forward(self, x):
        logits = []
        for model in self.models:
            with torch.no_grad():
                out = model(x)
                logits.append(out)
        avg_logits = torch.stack(logits).mean(dim=0)
        return avg_logits

    def idx_to_label(self, idx):
        if self.class_names is not None and idx < len(self.class_names):
            return self.class_names[idx]
        return str(idx)
