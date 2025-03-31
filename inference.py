#
# import os
# import torch
# import torchvision.transforms as transforms
# from torchvision.utils import save_image, make_grid
# from PIL import Image
# import argparse
#
# from train import DiffusionModel  # Import the DiffusionModel class from your training script
#
#
# def load_and_transform_image(image_path, transform):
#     """
#     Load an image and apply the specified transform
#
#     Args:
#         image_path (str): Path to the input image
#         transform (callable): Transformation to apply to the image
#
#     Returns:
#         torch.Tensor: Transformed image tensor
#     """
#     # Open image
#     img = Image.open(image_path).convert('RGB')
#
#     # Apply transform
#     transformed_img = transform(img)
#
#     return transformed_img
#
#
# def inference_single_image(
#         input_image_path,
#         output_dir,
#         checkpoint_name,
#         device=None,
#         dataset_name="custom_data",
#         timesteps=500,
#         beta1=1e-4,
#         beta2=0.02
# ):
#     """
#     Inference pipeline for generating a variation from a single input image
#     with side-by-side comparison
#
#     Args:
#         input_image_path (str): Path to the input image
#         output_dir (str): Directory to save generated images
#         checkpoint_name (str): Name of the checkpoint file
#         device (str, optional): Computing device
#         dataset_name (str, optional): Dataset name used during training
#         timesteps (int, optional): Number of diffusion timesteps
#         beta1 (float, optional): Starting noise parameter
#         beta2 (float, optional): Ending noise parameter
#     """
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Initialize diffusion model
#     diffusion_model = DiffusionModel(
#         device=device,
#         dataset_name=dataset_name,
#         checkpoint_name=checkpoint_name
#     )
#
#     # Get transforms used during training
#     transform, _ = diffusion_model.get_transforms(dataset_name)
#
#     # Load and transform input image
#     input_img = load_and_transform_image(input_image_path, transform)
#
#     # Prepare input batch for model
#     input_batch = input_img.unsqueeze(0).to(diffusion_model.device)
#
#     # Sample from the model
#     with torch.no_grad():
#         generated_samples, _, _ = diffusion_model.sample_ddpm(
#             n_samples=1,
#             context=input_batch,  # Use input image as context
#             timesteps=timesteps,
#             beta1=beta1,
#             beta2=beta2
#         )
#
#     # Get the generated sample
#     generated_sample = generated_samples[0]
#
#     # Prepare input and output images for comparison
#     # Convert from [-1, 1] to [0, 1] range
#     input_display = (input_img + 1) / 2
#     output_display = (generated_sample + 1) / 2
#
#     # Create side-by-side comparison
#     comparison_grid = make_grid([input_display, output_display], nrow=2)
#
#     # Create output filename based on input image name
#     input_filename = os.path.splitext(os.path.basename(input_image_path))[0]
#
#     # Save generated image
#     save_image(output_display, os.path.join(output_dir, f"{input_filename}_generated.png"))
#
#     # Save comparison grid
#     save_image(comparison_grid, os.path.join(output_dir, f"{input_filename}_comparison.png"))
#
#     print(f"Generated image saved to: {os.path.join(output_dir, f'{input_filename}_generated.png')}")
#     print(f"Comparison image saved to: {os.path.join(output_dir, f'{input_filename}_comparison.png')}")
#
#
# def main():
#     # Argument parsing
#     parser = argparse.ArgumentParser(description="Diffusion Model Single Image Variation Generation")
#     parser.add_argument("--input", required=True, help="Path to input image")
#     parser.add_argument("--output_dir", required=True, help="Directory to save generated images")
#     parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
#     parser.add_argument("--device", default=None, help="Computing device (cuda/cpu)")
#     parser.add_argument("--timesteps", type=int, default=500, help="Number of diffusion timesteps")
#
#     args = parser.parse_args()
#
#     # Run inference
#     inference_single_image(
#         input_image_path=args.input,
#         output_dir=args.output_dir,
#         checkpoint_name=args.checkpoint,
#         device=args.device,
#         timesteps=args.timesteps
#     )
#
#
# if __name__ == "__main__":
#     main()
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt

from train import DiffusionModel  # Import the DiffusionModel class from your training script


def debug_image_transforms(input_image_path, transform):
    """
    Debug the image transformation process

    Args:
        input_image_path (str): Path to the input image
        transform (callable): Transformation to apply to the image
    """
    # Open original image
    orig_img = Image.open(input_image_path).convert('RGB')

    # Convert to numpy for visualization
    orig_array = np.array(orig_img)

    # Apply transform
    transformed_img = transform(orig_img)

    # Visualize transformation steps
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(orig_array)

    plt.subplot(1, 3, 2)
    plt.title('Transformed Image (Tensor)')
    transformed_array = transformed_img.numpy().transpose(1, 2, 0)
    plt.imshow((transformed_array + 1) / 2)  # Denormalize for visualization

    plt.subplot(1, 3, 3)
    plt.title('Transformed Array Stats')
    plt.text(0.5, 0.5,
             f"Min: {transformed_img.min():.2f}\n"
             f"Max: {transformed_img.max():.2f}\n"
             f"Mean: {transformed_img.mean():.2f}",
             horizontalalignment='center',
             verticalalignment='center')

    plt.tight_layout()
    plt.show()

    return transformed_img


def load_and_transform_image(image_path, transform, debug=False):
    """
    Load an image and apply the specified transform

    Args:
        image_path (str): Path to the input image
        transform (callable): Transformation to apply to the image
        debug (bool): Whether to run debug visualization

    Returns:
        torch.Tensor: Transformed image tensor
    """
    # Open image
    img = Image.open(image_path).convert('RGB')

    # Apply transform
    transformed_img = transform(img)

    # Optional debug visualization
    if debug:
        debug_image_transforms(image_path, transform)

    return transformed_img


def inference_single_image(
        input_image_path,
        output_dir,
        checkpoint_name,
        device=None,
        dataset_name="custom_data",
        timesteps=500,
        beta1=1e-4,
        beta2=0.02,
        debug=False
):
    """
    Inference pipeline for generating a variation from a single input image
    with side-by-side comparison and optional debugging
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize diffusion model
    diffusion_model = DiffusionModel(
        device=device,
        dataset_name=dataset_name,
        checkpoint_name=checkpoint_name
    )

    # Get transforms used during training
    transform, _ = diffusion_model.get_transforms(dataset_name)

    # Load and transform input image
    input_img = load_and_transform_image(input_image_path, transform, debug)

    # Prepare input batch for model
    input_batch = input_img.unsqueeze(0).to(diffusion_model.device)

    # Sample from the model
    with torch.no_grad():
        generated_samples, _, _ = diffusion_model.sample_ddpm(
            n_samples=1,
            context=input_batch,  # Use input image as context
            timesteps=timesteps,
            beta1=beta1,
            beta2=beta2
        )

    # Get the generated sample
    generated_sample = generated_samples[0]

    # Create output filename based on input image name
    input_filename = os.path.splitext(os.path.basename(input_image_path))[0]

    # Save specific versions for debugging
    # 1. Raw generated sample (might have different range)
    save_image(generated_sample, os.path.join(output_dir, f"{input_filename}_raw_generated.png"))

    # Create side-by-side comparison
    input_display = (input_img + 1) / 2
    comparison_grid = make_grid([input_display, generated_sample], nrow=2)
    save_image(comparison_grid, os.path.join(output_dir, f"{input_filename}_comparison.png"))

    print(f"Generated images saved in {output_dir}")

    # Debug print for generated sample
    if debug:
        print("Generated Sample Stats:")
        print(f"Min: {generated_sample.min()}")
        print(f"Max: {generated_sample.max()}")
        print(f"Mean: {generated_sample.mean()}")


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Diffusion Model Single Image Variation Generation")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated images")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default=None, help="Computing device (cuda/cpu)")
    parser.add_argument("--timesteps", type=int, default=500, help="Number of diffusion timesteps")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")

    args = parser.parse_args()

    # Run inference
    inference_single_image(
        input_image_path=args.input,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint,
        device=args.device,
        timesteps=args.timesteps,
        debug=args.debug
    )


if __name__ == "__main__":
    main()