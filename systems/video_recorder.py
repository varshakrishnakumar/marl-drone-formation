import os
from typing import Optional

import imageio.v2 as imageio
import numpy as np


class VideoRecorder:
    """Simple video recorder that appends RGB frames and writes on close."""

    def __init__(self, path: str, fps: int = 30):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._writer: Optional[imageio.Writer] = imageio.get_writer(path, fps=fps)

    def add_frame(self, frame: np.ndarray) -> bool:
        """Append an RGB frame (H x W x 3, uint8) to the video.

        Returns False if the recorder has already been closed so callers can
        gracefully stop recording without raising.
        """
        if self._writer is None:
            return False

        self._writer.append_data(np.asarray(frame))
        return True

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


__all__ = ["VideoRecorder"]
