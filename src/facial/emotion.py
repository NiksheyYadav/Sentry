# Emotion Classification Module
# MobileNetV3-based emotion classifier with embedding extraction

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import densenet121, DenseNet121_Weights
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from ..config import FacialConfig


class LightingNormalization:
    """
    Custom transform for lighting normalization using CLAHE and gamma correction.
    
    This improves emotion recognition accuracy in low-light conditions by:
    1. Detecting if image is dark based on mean brightness
    2. Applying gamma correction to boost dark images
    3. Using CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast
    """
    
    def __init__(self, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize lighting normalization.
        
        Args:
            clip_limit: CLAHE clip limit for contrast limiting (higher = more contrast)
            tile_grid_size: Size of grid for adaptive histogram equalization
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        """
        Apply lighting normalization to an image.
        
        Args:
            img: PIL Image or numpy array (RGB or BGR)
            
        Returns:
            PIL Image with normalized lighting (grayscale)
        """
        # Convert PIL to numpy if needed
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        # Ensure we have a color image for processing
        if len(img_np.shape) == 2:
            # Already grayscale
            gray = img_np
        elif img_np.shape[2] == 4:
            # RGBA -> RGB -> Grayscale
            gray = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            # RGB -> Grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Check mean brightness to detect dark images
        mean_brightness = np.mean(gray)
        
        # Apply gamma correction for dark images (brightness < 100 on 0-255 scale)
        if mean_brightness < 100:
            # Calculate adaptive gamma based on brightness
            # Darker images get higher gamma (more brightening)
            gamma = 1.0 + (100 - mean_brightness) / 100  # Range: 1.0 to 2.0
            gamma = min(gamma, 1.8)  # Cap at 1.8 to avoid over-brightening
            
            # Apply gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                             for i in np.arange(0, 256)]).astype("uint8")
            gray = cv2.LUT(gray, table)
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(gray)
        
        # Convert back to PIL Image
        return Image.fromarray(enhanced)


@dataclass
class EmotionResult:
    """Emotion classification result."""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    embedding: np.ndarray  # 1280-dim embedding


class EmotionClassifier(nn.Module):
    """
    MobileNetV3-based emotion classifier.
    
    Classifies 7 basic emotions and extracts rich
    1280-dimensional embeddings for downstream fusion.
    """
    
    def __init__(self, config: Optional[FacialConfig] = None, pretrained: bool = True):
        """
        Initialize emotion classifier.
        
        Args:
            config: Facial analysis configuration.
            pretrained: Whether to use pretrained ImageNet weights.
        """
        super().__init__()
        self.config = config or FacialConfig()
        self.num_classes = len(self.config.emotion_classes)
        
        # Load pretrained MobileNetV3
        # if pretrained:
        #     weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        #     self.backbone = mobilenet_v3_large(weights=weights)
        # else:
        #     self.backbone = mobilenet_v3_large(weights=None)
        
        # Load pretrained DenseNet121
        if pretrained:
            weights = DenseNet121_Weights.IMAGENET1K_V1
            self.backbone = densenet121(weights=weights)
        else:
            self.backbone = densenet121(weights=None)
            
        # Modify first convolution for 1-channel input (Grayscale)
        # DenseNet121 features.conv0 is Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        old_conv = self.backbone.features.conv0
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Initialize 1-channel weights by averaging RGB weights
        if pretrained:
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        
        self.backbone.features.conv0 = new_conv
        
        # Get embedding dimension from backbone
        # self.embedding_dim = self.backbone.classifier[0].in_features  # 960 (MobileNetV3)
        self.embedding_dim = self.backbone.classifier.in_features  # 1024 (DenseNet121)
        
        # Replace classifier with emotion head
        self.backbone.classifier = nn.Identity()
        
        # Projection to match expected embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)  # Increased dropout for regularization
        )
        
        # Classification head with stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.config.embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, self.num_classes)
        )
        
        # Preprocessing with lighting normalization for low-light robustness
        # LightingNormalization applies CLAHE + gamma correction -> outputs grayscale PIL Image
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            LightingNormalization(clip_limit=3.0, tile_grid_size=(8, 8)),  # Adaptive lighting
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ])
        
        self._device = "cpu"
    
    def to(self, device):
        """Move model to device."""
        self._device = device
        return super().to(device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224).
            
        Returns:
            Tuple of (logits, embeddings).
        """
        # Get backbone features
        features = self.backbone(x)
        
        # Project to embedding space
        embeddings = self.projection(features)
        
        # Classify
        logits = self.classifier(embeddings)
        
        return logits, embeddings
    
    def predict(self, face_image: np.ndarray) -> EmotionResult:
        """
        Predict emotion for a face image.
        
        Args:
            face_image: RGB face image (HxWx3).
            
        Returns:
            EmotionResult with prediction and embedding.
        """
        self.eval()
        
        # Preprocess
        input_tensor = self.preprocess(face_image)
        input_tensor = input_tensor.unsqueeze(0).to(self._device)
        
        # Use FP16 if on CUDA
        use_amp = (self._device == "cuda" or (isinstance(self._device, torch.device) and self._device.type == "cuda"))
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, embedding = self.forward(input_tensor)
                probs = F.softmax(logits, dim=1)
        
        # Get prediction
        probs_np = probs.cpu().float().numpy()[0]
        pred_idx = np.argmax(probs_np)
        
        # Build probability dictionary
        prob_dict = {
            emotion: float(probs_np[i]) 
            for i, emotion in enumerate(self.config.emotion_classes)
        }
        
        # Refine probabilities to reduce noise/confusion
        prob_dict = self._refine_probabilities(prob_dict)
        
        # Update prediction based on refined probabilities
        pred_emotion = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[pred_emotion]
        
        return EmotionResult(
            emotion=pred_emotion,
            confidence=confidence,
            probabilities=prob_dict,
            embedding=embedding.cpu().numpy()[0]
        )
    
    def _refine_probabilities(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Refine probabilities using mutual exclusion and entropy-based noise rejection."""
        refined = probs.copy()
        
        # ===== EMERGENCY SAD OVERRIDE =====
        # The model has a severe Sad bias - it predicts Sad for Happy and Angry expressions
        # This override fixes the broken model predictions
        sad_prob = refined.get('sad', 0)
        happy_prob = refined.get('happy', 0)
        anger_prob = refined.get('anger', 0)
        fear_prob = refined.get('fear', 0)
        
        # ULTRA-AGGRESSIVE: If Sad is even moderately high (>0.5), check for overrides
        if sad_prob > 0.5:  # LOWERED from 0.7 to catch more cases
            # Check if Happy should actually win (smile detection)
            if happy_prob > 0.01:  # Even tiny Happy signal means user is smiling
                # OVERRIDE: User is smiling, not sad
                refined['happy'] = 0.85
                refined['sad'] = 0.05
                refined['neutral'] = 0.05
                # Suppress other emotions
                for k in refined:
                    if k not in ['happy', 'sad', 'neutral']:
                        refined[k] *= 0.1
            # Check if Anger should actually win (furrowed brows, tense face)
            # REMOVED THRESHOLD: ANY Anger presence means user is angry, not sad
            elif anger_prob > 0.001 or fear_prob > 0.005:  # Almost any signal
                # OVERRIDE: User is angry/tense, not sad
                refined['anger'] = 0.92  # MAXIMUM stability
                refined['sad'] = 0.01    # Almost eliminate Sad
                refined['neutral'] = 0.02
                refined['surprise'] = 0.01
                # Suppress other emotions completely
                for k in refined:
                    if k not in ['anger', 'sad', 'neutral', 'surprise']:
                        refined[k] *= 0.01
        
        # Calculate entropy (measure of uncertainty)
        # Higher entropy means the model is confused
        import math
        vals = [v for v in refined.values() if v > 0]
        entropy = -sum(v * math.log2(v) for v in vals)
        
        # 1. Entropy-Based Rejection: Aggressively reduced to allow emotions to surface
        # We no longer apply a special "distance" penalty to Anger
        happy_prob = refined.get('happy', 0)
        is_anger_present = refined.get('anger', 0) > 0.2 and happy_prob < 0.2
        
        if entropy > 1.85:
            # Minimal Neutral boost (Reduced to 0.05)
            boost = 0.02 if is_anger_present else 0.05 
            refined['neutral'] = refined.get('neutral', 0) + boost
            # Very light suppression (Increased to 0.85) to favor model predictions
            suppression = 0.9 if is_anger_present else 0.85
            for k in refined:
                if k != 'neutral':
                    # Protect happy if it's even slightly present
                    if k == 'happy' and happy_prob > 0.1:
                        continue
                    refined[k] *= suppression
        
        # 1b. Specific Anger Boost: Aggressive 1.5x boost
        if is_anger_present and refined['anger'] > refined.get('neutral', 0) * 0.5:
            refined['anger'] *= 1.5
        elif is_anger_present:
            # Even if weak, give it a fighting chance
            refined['anger'] *= 1.2
        
        # 1c. Specific Neutral Suppression: If negative emotions are trying to surface
        # but are being held back by a moderate neutral baseline.
        neg_emotions_raw = ['anger', 'fear', 'sad']
        max_neg_raw = max(refined.get(e, 0) for e in neg_emotions_raw)
        if max_neg_raw > 0.15:
            # Shift weight from neutral to the negative cluster
            shift = min(refined.get('neutral', 0), 0.2)
            refined['neutral'] -= shift
            # Give the shift to the strongest negative emotion
            winner_neg = max(neg_emotions_raw, key=lambda e: refined.get(e, 0))
            refined[winner_neg] += shift
        
        # 1d. NEUTRAL BOOST: If no strong emotions present, boost Neutral to prevent flickering
        all_emotions = ['happy', 'sad', 'anger', 'fear', 'surprise']
        max_emotion = max(refined.get(e, 0) for e in all_emotions)
        if max_emotion < 0.4:  # No strong emotion
            # User is likely neutral - boost it aggressively
            refined['neutral'] = 0.80
            # Suppress all other emotions
            for k in refined:
                if k != 'neutral':
                    refined[k] *= 0.1
            
        # 2. Mutual Exclusion: Happy vs Surprise vs Negative
        happy_prob = refined.get('happy', 0)
        surprise_prob = refined.get('surprise', 0)
        
        # 2a. Happy vs Surprise Disentanglement - REFINED
        # Only override if Surprise is VERY weak (< 0.05) and Happy is strong
        # This prevents false Surprise detection
        if surprise_prob > 0.01 and surprise_prob < 0.05 and happy_prob > 0.6:
            # Surprise is very weak, Happy is strong - likely a smile, not surprise
            refined['surprise'] = 0.02
            refined['happy'] = 0.85
        elif surprise_prob > 0.05 and happy_prob > 0.5:
            # Surprise is stronger - likely genuine surprise (open mouth)
            refined['surprise'] = 0.80
            refined['happy'] = 0.10
        elif happy_prob > 0.35 and surprise_prob > 0.1:
            # If smiling, it's probably not "Surprise"
            refined['surprise'] *= 0.4
        elif surprise_prob > 0.35 and happy_prob > 0.1:
            # If surprised, it's rarely "Happy" at the same time
            if happy_prob < 0.5:
                refined['happy'] *= 0.5
                
        # 2b. Happy vs Negative Emotions - FIXED LOGIC
        negative_emotions = ['anger', 'fear', 'sad']
        max_neg = max(refined.get(e, 0) for e in negative_emotions)
        
        # Protect Happy if it's clearly the strongest emotion
        if happy_prob > 0.35 and happy_prob > max_neg:
            # Happy is genuinely winning - suppress negative emotions
            suppression = 0.1 if happy_prob > 0.6 else 0.2
            for emo in negative_emotions:
                if emo in refined:
                    refined[emo] *= suppression
        elif happy_prob > 0.4:
            # Happy is strong but not dominant - still suppress negatives moderately
            for emo in negative_emotions:
                if emo in refined:
                    refined[emo] *= 0.3
        elif max_neg > happy_prob and max_neg > 0.3:
            # A SINGLE negative emotion is genuinely stronger than Happy
            # Only then suppress Happy
            refined['happy'] = refined.get('happy', 0) * 0.2
            refined['neutral'] *= 0.3
            
        # 3. Specific Confusion: Anger vs Fear vs Sad
        # These are often confused; we use a "Winner-Takes-All" strategy
        # to prevent flickering within the negative cluster.
        # BUT: Don't apply this if Happy is clearly present (prevents Happy→Sad misclassification)
        cluster_emotions = ['anger', 'fear', 'sad']
        cluster_probs = {e: refined.get(e, 0) for e in cluster_emotions}
        happy_prob_current = refined.get('happy', 0)
        
        # Only apply Winner-Takes-All if Happy is NOT dominant
        if sum(cluster_probs.values()) > 0.4 and happy_prob_current < 0.3:
            winner = max(cluster_probs, key=cluster_probs.get)
            win_val = cluster_probs[winner]
            
            # If there's a clear candidate or collective strength
            if win_val > 0.25:
                # Polarize: Boost the winner and suppress others in the cluster
                for emo in cluster_emotions:
                    if emo == winner:
                        refined[emo] *= 1.25 # Boost the winner
                    else:
                        refined[emo] *= 0.15 # Heavily suppress others
            
        # 4. Final Normalization
        total = sum(refined.values())
        if total > 0:
            refined = {k: v / total for k, v in refined.items()}
            
        return refined
    
    def predict_batch(self, face_images: List[np.ndarray]) -> List[EmotionResult]:
        """
        Predict emotions for multiple face images.
        
        Args:
            face_images: List of RGB face images.
            
        Returns:
            List of EmotionResult objects.
        """
        if not face_images:
            return []
        
        self.eval()
        
        # Preprocess all images
        inputs = torch.stack([
            self.preprocess(img) for img in face_images
        ]).to(self._device)
        
        with torch.no_grad():
            logits, embeddings = self.forward(inputs)
            probs = F.softmax(logits, dim=1)
        
        # Build results
        probs_np = probs.cpu().numpy()
        embeddings_np = embeddings.cpu().numpy()
        
        results = []
        for i in range(len(face_images)):
            pred_idx = np.argmax(probs_np[i])
            prob_dict = {
                emotion: float(probs_np[i, j])
                for j, emotion in enumerate(self.config.emotion_classes)
            }
            
            results.append(EmotionResult(
                emotion=self.config.emotion_classes[pred_idx],
                confidence=float(probs_np[i, pred_idx]),
                probabilities=prob_dict,
                embedding=embeddings_np[i]
            ))
        
        return results
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract only the embedding for a face.
        
        Args:
            face_image: RGB face image.
            
        Returns:
            1280-dimensional embedding.
        """
        result = self.predict(face_image)
        return result.embedding


def create_emotion_classifier(config: Optional[FacialConfig] = None, 
                              device: str = "cuda") -> EmotionClassifier:
    """
    Factory function to create emotion classifier.
    
    Args:
        config: Facial analysis configuration.
        device: Computation device.
        
    Returns:
        Initialized EmotionClassifier on specified device.
    """
    device = device if torch.cuda.is_available() else "cpu"
    classifier = EmotionClassifier(config=config)
    classifier.to(device)
    return classifier
