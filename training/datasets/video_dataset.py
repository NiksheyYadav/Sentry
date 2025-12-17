# Custom Video Dataset
# For self-collected mental health assessment data

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import json


class CustomVideoDataset(Dataset):
    """
    Dataset for self-collected video sessions with per-frame labels.
    
    Expects data structure:
    data/custom_sessions/
    ├── session_001/
    │   ├── video.mp4 (or frames/)
    │   └── labels.json
    ├── session_002/
    ...
    
    labels.json format:
    {
        "stress_labels": [1, 1, 2, 2, 3, ...],  # Per-second labels 1-5
        "notes": "Study session, exam prep",
        "subject_id": "anonymous_001"
    }
    """
    
    STRESS_LEVELS = ['low', 'mild', 'moderate', 'high', 'severe']
    
    def __init__(
        self,
        root_dir: str,
        frames_per_clip: int = 30,
        frame_stride: int = 10,
        transform=None
    ):
        """
        Initialize video dataset.
        
        Args:
            root_dir: Path to sessions directory
            frames_per_clip: Number of frames per sample
            frame_stride: Stride between sampled frames
            transform: Image transforms
        """
        self.root_dir = Path(root_dir)
        self.frames_per_clip = frames_per_clip
        self.frame_stride = frame_stride
        self.transform = transform
        
        # Load all sessions
        self.clips = self._load_clips()
    
    def _load_clips(self) -> List[Dict]:
        """Load all clips from sessions."""
        clips = []
        
        for session_dir in self.root_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            labels_path = session_dir / 'labels.json'
            video_path = session_dir / 'video.mp4'
            frames_dir = session_dir / 'frames'
            
            if not labels_path.exists():
                print(f"Skipping {session_dir}: no labels.json")
                continue
            
            # Load labels
            with open(labels_path) as f:
                labels = json.load(f)
            
            stress_labels = labels.get('stress_labels', [])
            
            # Determine source (video or frames)
            if video_path.exists():
                source = ('video', video_path)
            elif frames_dir.exists():
                source = ('frames', frames_dir)
            else:
                print(f"Skipping {session_dir}: no video or frames")
                continue
            
            # Create clips (one per second of video)
            for sec_idx, stress_label in enumerate(stress_labels):
                clips.append({
                    'session': session_dir.name,
                    'source': source,
                    'second': sec_idx,
                    'stress_label': stress_label - 1,  # 0-indexed (0-4)
                    'subject': labels.get('subject_id', 'unknown')
                })
        
        print(f"CustomVideoDataset: Loaded {len(clips)} clips from {self.root_dir}")
        return clips
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        clip_info = self.clips[idx]
        source_type, source_path = clip_info['source']
        second = clip_info['second']
        
        # Extract frames for this second
        if source_type == 'video':
            frames = self._extract_frames_from_video(
                source_path, second, self.frames_per_clip
            )
        else:
            frames = self._load_frames_from_dir(
                source_path, second, self.frames_per_clip
            )
        
        # Apply transforms to each frame
        if self.transform:
            frames = [self.transform(Image.fromarray(f)) for f in frames]
        else:
            frames = [
                torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                for f in frames
            ]
        
        # Stack into (C, T, H, W) format
        video_tensor = torch.stack(frames, dim=1)
        
        return video_tensor, clip_info['stress_label']
    
    def _extract_frames_from_video(
        self, video_path: Path, second: int, num_frames: int
    ) -> List[np.ndarray]:
        """Extract frames from video file."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        frames = []
        start_frame = int(second * fps)
        
        for i in range(num_frames):
            frame_idx = start_frame + i * (fps // num_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                # Pad with last frame or black
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        return frames
    
    def _load_frames_from_dir(
        self, frames_dir: Path, second: int, num_frames: int
    ) -> List[np.ndarray]:
        """Load frames from directory of extracted frames."""
        frames = []
        
        # Assume frames named like frame_0001.jpg
        for i in range(num_frames):
            frame_idx = second * 30 + i * (30 // num_frames)
            frame_path = frames_dir / f'frame_{frame_idx:04d}.jpg'
            
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames
    
    @staticmethod
    def create_session_template(output_dir: str, num_seconds: int = 60) -> None:
        """
        Create a template session directory for labeling.
        
        Args:
            output_dir: Path to create session
            num_seconds: Expected video duration
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Create labels template
        labels = {
            "stress_labels": [1] * num_seconds,  # 1-5 scale
            "notes": "Add session notes here",
            "subject_id": "anonymous_001"
        }
        
        with open(out_path / 'labels.json', 'w') as f:
            json.dump(labels, f, indent=2)
        
        print(f"Created session template at: {out_path}")
        print("Instructions:")
        print("1. Record a video and save as 'video.mp4' in this folder")
        print("2. Edit labels.json with per-second stress labels (1-5)")
        print("   1=calm, 2=slightly stressed, 3=moderate, 4=high, 5=severe")
