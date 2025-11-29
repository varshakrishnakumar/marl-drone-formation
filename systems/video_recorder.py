import os
from typing import Optional

import imageio.v2 as imageio
import numpy as np


class VideoRecorder:
    def __init__(self, path: str, fps: int = 30):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._writer: Optional[imageio.Writer] = imageio.get_writer(path, fps=fps)

    def add_frame(self, frame: np.ndarray) -> None:
        """Append an RGB frame (H x W x 3, uint8) to the video."""
        if self._writer is None:
            raise RuntimeError("VideoRecorder is closed")
        self._writer.append_data(np.asarray(frame))
        
    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None  


__all__ = ["VideoRecorder"]
