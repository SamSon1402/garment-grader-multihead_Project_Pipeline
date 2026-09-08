import numpy as np

from garment_grader.core.schemas import Box
from garment_grader.projector.geometry import ProjectorCalibration
from garment_grader.projector.sync import projection_delay_ms


def test_projection_delay_subtracts_processing_time():
    # 1.8m / 0.45m/s = 4s; 100ms already spent -> 3900ms remaining
    assert projection_delay_ms(1.8, 0.45, 100) == 3900


def test_identity_homography_keeps_box():
    cal = ProjectorCalibration(np.eye(3, dtype=np.float32))
    box = cal.box(Box(x1=10, y1=20, x2=30, y2=40))
    assert box.x1 == 10
    assert box.y2 == 40
