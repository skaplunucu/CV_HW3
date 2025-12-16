import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from typing import List, Tuple, Optional
import torch

DATA_DIR = Path('../../data/CelebAMask-HQ')
CELEBA_IMG_DIR = DATA_DIR / 'CelebA-HQ-img'
RESULTS_DIR = Path('../../Report/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_face_image(image_id: int, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    img_path = CELEBA_IMG_DIR / f"{image_id}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img


def load_face_pair(id1: int = 0, id2: int = 100, target_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, np.ndarray]:
    img1 = load_face_image(id1, target_size)
    img2 = load_face_image(id2, target_size)
    return img1, img2


def visualize_image_pair(img1: np.ndarray, img2: np.ndarray, title1: str = "Image 1", title2: str = "Image 2"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img1)
    axes[0].set_title(title1, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].set_title(title2, fontweight='bold')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def calculate_rmse(pts1: np.ndarray, pts2: np.ndarray) -> float:
    if pts1.shape != pts2.shape:
        raise ValueError(f"Point sets must have same shape: {pts1.shape} vs {pts2.shape}")
    squared_diff = np.sum((pts1 - pts2) ** 2, axis=1)
    rmse = np.sqrt(np.mean(squared_diff))
    return rmse


def calculate_alignment_rmse(pts1: np.ndarray, pts2: np.ndarray, transform_matrix: np.ndarray) -> float:
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    if transform_matrix.shape == (3, 3):
        pts1_transformed_h = (transform_matrix @ pts1_h.T).T
        pts1_transformed = pts1_transformed_h[:, :2] / pts1_transformed_h[:, 2:3]
    elif transform_matrix.shape == (2, 3):
        pts1_transformed = (transform_matrix @ pts1_h.T).T
    else:
        raise ValueError(f"Invalid transform matrix shape: {transform_matrix.shape}")
    rmse = calculate_rmse(pts1_transformed, pts2)
    return rmse


def save_as_gif(frames: List[np.ndarray], output_path: Path, fps: int = 10, loop: int = 0):
    frames_uint8 = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames_uint8.append(frame)
    imageio.mimsave(output_path, frames_uint8, fps=fps, loop=loop)


def save_as_mp4(frames: List[np.ndarray], output_path: Path, fps: int = 30):
    if len(frames) == 0:
        raise ValueError("No frames to save")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), radius: int = 2) -> np.ndarray:
    img_copy = image.copy()
    for x, y in landmarks:
        cv2.circle(img_copy, (int(x), int(y)), radius, color, -1)
    return img_copy


def draw_matches_custom(img1: np.ndarray, kp1: List, img2: np.ndarray, kp2: List,
                        matches: List, max_matches: int = 50) -> np.ndarray:
    matches_to_draw = matches[:max_matches]
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches_to_draw,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return match_img
