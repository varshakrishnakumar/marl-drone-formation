import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, path, fps=30):
        self.path = path
        self.fps = fps
        self.frames = []

    def add_frame(self, img_rgb):
        """
        img_rgb: (H,W,3) numpy array, uint8
        """
        self.frames.append(img_rgb)

    def save(self):
        print(f"Saving video to {self.path} ...")
        imageio.mimwrite(self.path, self.frames, fps=self.fps)
        print("Video saved.")
