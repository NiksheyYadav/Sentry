# Data Augmentation Transforms
# Training and validation transforms for emotion recognition

from torchvision import transforms
from typing import Tuple


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    horizontal_flip: bool = True,
    rotation: int = 15,
    color_jitter: bool = True
) -> transforms.Compose:
    """
    Get training transforms with data augmentation.
    
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
        transform_list.append(transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ))
    
    transform_list.extend([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
    
    return transforms.Compose(transform_list)


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
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
