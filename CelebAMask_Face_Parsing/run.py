import argparse
import cv2
import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image

# Import your model
from mobilenet_unet_cbam import MobileNet_UNet_CBAM

def main(input, output, weights):
    print(input)
    print(output)
    # Load the input image
    image = cv2.imread(input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))  # Resize to match model input

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Initialize the model
    model = MobileNet_UNet_CBAM()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
        
    ckpt = torch.load(weights, map_location=torch.device('cpu'))
    # NOTE: Make sure that the weights are saved in the "state_dict" key
    # DO NOT CHANGE THIS VALUE, i.e., ckpt["state_dict"]
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Perform inference
    with torch.no_grad():
        image = image.to(device)
        prediction = model(image)
        pred_mask = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
    
    # Convert prediction to an image
    mask_img = Image.fromarray(pred_mask.astype(np.uint8))

    # Define a color palette for segmentation masks
    palette = [0, 0, 0, 204, 0, 0, 76, 153, 0, 204, 204, 0, 51, 51, 255, 204, 0, 204, 0, 255, 255, 255, 204, 204, 102, 51, 0, 255, 0, 0, 102, 204, 0, 255, 255, 0, 0, 0, 153, 0, 0, 204, 255, 51, 153, 0, 204, 204, 0, 51, 0, 255, 153, 51, 0, 204, 0]
    mask_img.putpalette(palette)

    # Save the predicted mask
    mask_img.save(output)
    print(f"Saved prediction mask at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--weights", type=str, default="ckpt.pth")
    args = parser.parse_args()
    main(args.input, args.output, args.weights)