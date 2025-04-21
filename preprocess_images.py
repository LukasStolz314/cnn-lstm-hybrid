import os
import cv2
import numpy as np
import mediapipe as mp
from settings import *

MOUTH_INDICES = [0, 267, 269, 270, 409, 306, 375, 321, 405, 314,
                 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]

def crop_mouth_from_video(video_path, output_path, crop_size=(100, 70), max_frames=30, padding=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                xs, ys = [], []

                for idx in MOUTH_INDICES:
                    lm = landmarks[idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    xs.append(x)
                    ys.append(y)

                x1, x2 = max(0, min(xs)), min(w, max(xs))
                y1, y2 = max(0, min(ys)), min(h, max(ys))

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
            else:
                x1, y1, x2, y2 = w // 2 - 100, h // 2 - 50, w // 2 + 100, h // 2 + 50

            cropped = frame[y1:y2, x1:x2]
            try:
                cropped = cv2.resize(cropped, crop_size)
            except:
                cropped = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)

            frames.append(cropped)

    cap.release()

    while len(frames) < max_frames:
        frames.append(np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8))

    frames_array = np.array(frames)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, frames_array)
    print(f"Saved: {output_path}")

def view_npy_video(npy_path, fps=20):
    frames = np.load(npy_path)

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Mouth Video", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def compute_mean_std_from_npy(directory):
    sum_ = np.zeros(3)
    sum_squared = np.zeros(3)
    pixel_count = 0

    file_list = [f for f in os.listdir(directory) if f.endswith(".npy")]

    for file in file_list:
        frames = np.load(os.path.join(directory, file))
        frames = frames.astype(np.float32) / 255.0

        flat = frames.reshape(-1, 3)
        sum_ += flat.sum(axis=0)
        sum_squared += (flat ** 2).sum(axis=0)
        pixel_count += flat.shape[0]

    mean = sum_ / pixel_count
    std = np.sqrt((sum_squared / pixel_count) - (mean ** 2))
    return mean, std


if __name__ == "__main__":
    import glob

    

    video_paths = glob.glob(os.path.join(base_dir, "*.mp4"))

    for label in corpus:
        label_path = os.path.join(base_dir, label)
        if os.path.isdir(label_path):
                for video_file in os.listdir(os.path.join(label_path, 'train')):
                    # print(video_file)
                    if video_file.lower().endswith('.mp4'):
                        video_path = os.path.join(label_path, 'train', video_file)
                        output_path = os.path.join(processed_img_dir, video_file + ".npy")
                        crop_mouth_from_video(video_path, output_path)

    # for video in os.listdir(processed_img_dir)[:1]:
    #     video_path = os.path.join(processed_img_dir, video)
    #     view_npy_video(video_path)

    mean, std = compute_mean_std_from_npy(processed_img_dir)
    np.save(mean_std_file, {'mean': mean, 'std': std})

