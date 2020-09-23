from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .middlebury_dataset import MiddleburyDatset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "middlebury": MiddleburyDatset
}
