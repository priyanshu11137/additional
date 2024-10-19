import torch
from models import vgg19
import gdown
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import os
from typing import Union, Tuple
from pathlib import Path

class CrowdCounter:
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize the CrowdCounter with either a local model path or Google Drive URL.
        
        Args:
            model_path: Either a local file path or a Google Drive URL
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = self._prepare_model_path(model_path)
        self.model = self._load_model()
        
    def _prepare_model_path(self, model_path: Union[str, Path]) -> str:
        """
        Handles both local paths and Google Drive URLs.
        Downloads the model if it's a Google Drive URL.
        """
        # Check if input is a Google Drive URL
        if str(model_path).startswith('https://drive.google.com'):
            local_path = "pretrained_models/model.pth"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            gdown.download(str(model_path), local_path, quiet=False)
            return local_path
        else:
            # Ensure the local path exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            return str(model_path)
    
    def _load_model(self) -> torch.nn.Module:
        """
        Loads the model from the prepared path.
        """
        model = vgg19()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        return model
    
    def _preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the input image for model inference.
        
        Args:
            image: Can be a file path, numpy array, or PIL Image
        """
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype('uint8'), 'RGB')
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError("Unsupported image format")
            
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return transform(img).unsqueeze(0).to(self.device)
    
    def process_image(self, image: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, int]:
        """
        Process an image and return the density map and count.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            Tuple containing:
            - Visualization of the density map (numpy array)
            - Predicted count (integer)
        """
        # Preprocess and run inference
        input_tensor = self._preprocess_image(image)
        
        with torch.no_grad():
            outputs, _ = self.model(input_tensor)
        
        count = int(torch.sum(outputs).item())
        
        # Generate density map visualization
        density_map = outputs[0, 0].cpu().numpy()
        normalized_map = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
        vis_map = (normalized_map * 255).astype(np.uint8)
        colored_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
        rgb_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)
        
        return rgb_map, count

def main():
    # Example usage
    # Local model
    image_path=r"C:\Users\prj11\Downloads\EE798r\DM-Count\example_images\44.jpg"
    try:
        counter_local = CrowdCounter(r"C:\Users\prj11\Downloads\EE798r\DM-Count\pretrained_models\model_qnrf.pth")
        result_map, count = counter_local.process_image(image_path)
        print(f"Local model count: {count}")
    except FileNotFoundError:
        print("Local model file not found")
    
    # Google Drive model
    gdrive_url = "https://drive.google.com/uc?id=1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru"
    counter_gdrive = CrowdCounter(gdrive_url)
    result_map, count = counter_gdrive.process_image(image_path)
    print(f"Google Drive model count: {count}")

if __name__ == "__main__":
    main()