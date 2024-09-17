from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS

@DATASETS.register_module()
class BingRGBDataset(CustomDataset):
    """BingRGB LULC dataset."""
    
    CLASSES = (
        'background', 'farmland', 'water', 'forest', 'urban_structure', 
        'rural_built_up', 'urban_built_up', 'road', 'meadow', 'marshland', 'brick_factory'
    )
    
    PALETTE = [
        [0, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [128, 0, 0],
        [255, 0, 255], [255, 0, 0], [160, 160, 164], [255, 255, 0], 
        [255, 251, 240], [128, 0, 128]
    ]
    
    def __init__(self, **kwargs):
        # Remove `reduce_zero_label` here since it will be provided by the configuration
        super(BingRGBDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            **kwargs)  # **kwargs will take care of other arguments from the config
