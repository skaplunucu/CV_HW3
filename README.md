# CV2025 Homework 3: Face Segmentation and Matching

**December 2025**

## Assignment

### Datasets

- **CelebAMaskHQ** — https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
- **Multiclass Face Segmentation** — https://www.kaggle.com/datasets/ashish2001/multiclass-face-segmentation

### Tools

Google Colab, Python, OpenCV, PyTorch, YOLO-Seg, MediaPipe, SAM3

**Estimated time:** 16 hours

---

## Task Overview

This assignment consists of two parts: (1) face segmentation using classical and modern methods, and (2) face matching using keypoints and MediaPipe, creating a morph animation. Your implementation should include quantitative evaluation and visual comparisons.

---

## Part 1: Face Segmentation

Your task is to segment facial parts on a portrait image (hair, mouth, nose, eyes, ears, and neck) using both classical and neural-network methods. Your report should contain table with comparison of all methods (method, dataset, inference Time, IoU, F1 score).

1. **Classical Segmentation** — Convert images to HSV and apply GMM model for colors and locations of different face parts.

2. **YOLO-Seg Pretraining** — Use a small subset (50–100 images) to fine-tune a YOLO-Seg (one of your choice) model to obtain face segmentation masks.

3. **Ready Face Segmentation Model** — Apply an existing pretrained face-segmentation model (e.g., BiSeNet).

4. **SAM3 Inference** — Adopt and run SAM3 on the same test images. Measure segmentation quality and visualize overlays.

5. **Comparison and Summary** — Create a table with metrics (method, dataset, inference Time, IoU, F1 score, examples). Include visual examples and a short written comparison of all four methods.

---

## Part 2: Face Matching and Morphing

1. **Keypoint-Based Matching** — Detect ORB/SIFT keypoints on two aligned face images. Match descriptors, filter inliers with RANSAC, and visualize matched pairs.

2. **MediaPipe-Based Matching** — Extract MediaPipe Face Mesh landmarks. Obtain dense optical flow from landmarks.

3. **Face Morph Animation** — Using the MediaPipe correspondences, implement a simple face morph: interpolate landmarks over N frames and warp images accordingly. Export a short GIF or MP4 showing the morph between two portraits.

4. **Evaluation and Export** — Compare classical vs MediaPipe matching via alignment RMSE. Include the generated morph animation as part of the deliverables.

---

## Deliverables

- Colab notebook with code for segmentation and matching.
- Metrics table (IoU, Dice, RMSE).
- Visual results for all segmentation methods.
- Morph animation (GIF or MP4).
- Short summary (0.5–1 page).

---

## Attached Files

- [CV2025_M3_Homework.pdf](Report/CV2025_M3_Homework.pdf)