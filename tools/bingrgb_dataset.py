from mmseg.datasets import CustomDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class BingRGBDataset(CustomDataset):
    """BingRGB LULC dataset."""
    
    # These are the class names and background that correspond to the class map
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
        super(BingRGBDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            reduce_zero_label=True,
            **kwargs)
