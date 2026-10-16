import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """10 s, 10 fps, 320x240 video whose frame index is encoded in the pixel intensity."""
    path = tmp_path_factory.mktemp("vid") / "synthetic.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for i in range(100):
        frame = np.full((240, 320, 3), (i * 2) % 256, dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        w.write(frame)
    w.release()
    return str(path)
