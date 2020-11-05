from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .middlebury_dataset import MiddleburyDatset
from .scared_dataset import ScaredDatset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "middlebury": MiddleburyDatset,
    "scared": ScaredDatset
}
