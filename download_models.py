import os
import gdown

os.makedirs("saved_models", exist_ok=True)

models = {
    "best_vgg_model.pth":"1t7Pr4d6rGYotQwZyMC31lm0P4mx1GtkP",
    "best_resnet50_model_3_classes_combined.pth":"1yqAP-c2wN_wdazvc-aSUx83A3WjPmFiH"
}

for filename, file_id in models.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output = os.path.join("saved_models", filename)
    if not os.path.exists(output):
        print(f"Downloading {filename}...")
        gdown.download(url, output, quiet=False)
    else:
        print(f"{filename} already exists, skipping download.")
