import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from models import vgg19
import multiprocessing
import cv2

def main():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--device', default='0', help='assign device')  # Default GPU device is 0
    parser.add_argument('--use-cpu', action='store_true', help='force use of CPU even if GPU is available')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--model-path', type=str, default='pretrained_models/model_qnrf.pth',
                        help='saved model path')
    parser.add_argument('--data-path', type=str,
                        default='data/QNRF-Train-Val-Test',
                        help='data path for testing images')
    parser.add_argument('--dataset', type=str, default='qnrf',
                        help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--pred-density-map-path', type=str, default='',
                        help='save predicted density maps when pred-density-map-path is not empty.')

    args = parser.parse_args()

    # Set device based on whether CPU is forced or if GPU is available
    if args.use_cpu:
        device = torch.device('cpu')
        print("Running on CPU.")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # Set CUDA device if not using CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Running on GPU: CUDA device {args.device}.")
        else:
            print("GPU not available, running on CPU.")

    # Clear CUDA cache to avoid memory leakage (only if using GPU)
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path

    # Dataset loading based on the selected dataset
    if args.dataset.lower() == 'qnrf':
        dataset = crowd.Crowd_qnrf(os.path.join(data_path, 'test'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'nwpu':
        dataset = crowd.Crowd_nwpu(os.path.join(data_path, 'val'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
        dataset = crowd.Crowd_sh(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported!")

    # Create dataloader with efficient num_workers
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1,  # Ensure batch size is 1 for evaluation
        shuffle=False, 
        num_workers=min(multiprocessing.cpu_count(), 4), 
        pin_memory=True if device.type == 'cuda' else False  # Pin memory if using GPU
    )

    # Create the output directory for predicted density maps if specified
    if args.pred_density_map_path:
        if not os.path.exists(args.pred_density_map_path):
            os.makedirs(args.pred_density_map_path)

    def load_model(model, model_path, device):
        """ Load model from checkpoint. """
        checkpoint = torch.load(model_path, map_location=device)  # Load on the selected device
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model

    # Load the VGG19 model
    model = vgg19()
    model = load_model(model, model_path, device)
    model.to(device)
    model.eval()

    image_errs = []
    
    # Iterate over the test set
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'Batch size should be 1 for evaluation.'

        with torch.no_grad():  # Disable gradient computation to save memory during evaluation
            outputs, _ = model(inputs)
        
        # Compute error
        img_err = count[0].item() - torch.sum(outputs).item()
        print(f"Image: {name}, Error: {img_err}, Actual Count: {count[0].item()}, Predicted Count: {torch.sum(outputs).item()}")
        image_errs.append(img_err)

        # Save predicted density map if path is provided
        if args.pred_density_map_path:
            vis_img = outputs[0, 0].cpu().numpy()
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.png'), vis_img)

    # Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print(f"Model Path: {model_path}\nMAE: {mae}, MSE: {mse}\n")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Necessary for Windows multiprocessing
    main()
