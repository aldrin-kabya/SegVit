from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS

@DATASETS.register_module()
class BingRGBDataset(CustomDataset):
    """BingRGB LULC dataset."""
    
    # # Original 11 classes and pallete
    # CLASSES = (
    #     'background', 'farmland', 'water', 'forest', 'urban_structure', 
    #     'rural_built_up', 'urban_built_up', 'road', 'meadow', 'marshland', 'brick_factory'
    # )
    
    # PALETTE = [
    #     [0, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [128, 0, 0],
    #     [255, 0, 255], [255, 0, 0], [160, 160, 164], [255, 255, 0], 
    #     [255, 251, 240], [128, 0, 128]
    # ]

    # 1st selected classes and pallete
    CLASSES = (
        'background', 'farmland', 'water', 'forest', 
        'urban_structure', 'meadow'
    )
    
    PALETTE = [
        [0, 0, 0],    # background
        [0, 255, 0],  # farmland
        [0, 0, 255],  # water
        [0, 255, 255],# forest
        [128, 0, 0],  # urban_structure (merged with rural_built_up, urban_built_up, road, and brick_factory)
        [255, 255, 0] # meadow (merged with marshland)
    ]

    # # BDSAT Paper classes and pallete
    # CLASSES = (
    #     'background', 'forest', 'urban_structure', 'water', 
    #     'farmland', 'meadow'
    # )
    
    # PALETTE = [
    #     [0, 0, 0],    # background
    #     [0, 255, 255],# forest
    #     [128, 0, 0],  # urban_structure (merged with rural_built_up, urban_built_up, road, and brick_factory)
    #     [0, 0, 255],  # water
    #     [0, 255, 0],  # farmland
    #     [255, 255, 0] # meadow (merged with marshland)
    # ]
    
    def __init__(self, **kwargs):
        # Remove `reduce_zero_label` here since it will be provided by the configuration
        super(BingRGBDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)  # **kwargs will take care of other arguments from the config
