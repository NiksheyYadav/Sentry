# Data Augmentation Transforms
# Training and validation transforms for emotion recognition

from torchvision import transforms
from typing import Tuple


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    horizontal_flip: bool = True,
    rotation: int = 20,  # Increased rotation
    color_jitter: bool = True
) -> transforms.Compose:
    """
    Get training transforms with STRONG data augmentation to reduce overfitting.
    
    Args:
        image_size: Output image size (height, width)
        horizontal_flip: Enable random horizontal flipping
        rotation: Max rotation angle in degrees
        color_jitter: Enable color jittering
        
    Returns:
        Composed training transforms
    """
    transform_list = [
        transforms.Resize((image_size[0] + 32, image_size[1] + 32)),
        transforms.RandomCrop(image_size),
    ]
    
    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))
    
    if color_jitter:
        # More aggressive color jitter for better generalization
        transform_list.append(transforms.ColorJitter(
            brightness=0.3,  # Increased from 0.2
            contrast=0.3,    # Increased from 0.2
            saturation=0.3,  # Increased from 0.2
            hue=0.15         # Increased from 0.1
        ))
    
    transform_list.extend([
        transforms.RandomAffine(
            degrees=10,       # Added rotation here too
            translate=(0.15, 0.15),  # Increased from 0.1
            scale=(0.85, 1.15),      # Increased range
            shear=10                  # Added shear
        ),
        # Add Gaussian blur for robustness
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
        # Add random perspective for more variation
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        ),
        # More aggressive random erasing (cutout regularization)
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])
    
    return transforms.Compose(transform_list)


def get_aggressive_transforms(
    image_size: Tuple[int, int] = (224, 224)
) -> transforms.Compose:
    """
    Get VERY aggressive transforms for oversampled minority class images.
    
    Use this with balanced datasets to maximize diversity of augmented samples
    from underrepresented classes like sad and neutral.
    
    Args:
        image_size: Output image size (height, width)
        
    Returns:
        Composed aggressive training transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size[0] + 48, image_size[1] + 48)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),  # Stronger rotation
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2
        ),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.2, 0.2),
            scale=(0.8, 1.2),
            shear=15
        ),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))], p=0.3),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        transforms.RandomAutocontrast(p=0.2),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(0.3, 3.3))
    ])


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224)
) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Output image size
        
    Returns:
        Composed validation transforms
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
