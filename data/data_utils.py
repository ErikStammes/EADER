import torch
import numpy as np

class BaseSegmentationMapToClass():
    """ Preprocesses the input image by setting the void pixels to background pixels (255 > 0) """
    def __call__(self, image):
        image = np.array(image, dtype=np.uint8)
        image = np.where(image == 255, 0, image)
        return image

class SegmentationMapToClass(BaseSegmentationMapToClass):
    """ Extract the class of a segmentation map based on the largest connected component. """
    def __call__(self, image):
        image = super().__call__(image)
        largest_connected_component = image == np.argmax(np.bincount(image.flatten())[1:]) + 1
        largest_class = np.unique(image[largest_connected_component])
        assert len(largest_class) == 1
        return torch.as_tensor(largest_class[0], dtype=torch.int64)

class SegmentationMapToMultiClass(BaseSegmentationMapToClass):
    """ Extracts all classes (except background) of a segmentation map and return a one-hot encoding. """
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, image):
        image = super().__call__(image)
        classes = np.unique(image)
        classes = classes[classes != 0]
        result = np.zeros(self.num_classes)
        result[classes - 1] = 1
        assert np.any(result)
        return torch.as_tensor(result, dtype=torch.float32)

class SegmentationMapAndMultiClass(SegmentationMapToMultiClass):
    """ Extracts all classes (except background) of a segmentation map and returns both a one-hot encoding
        and the original segmentation map. """
    def __call__(self, image):
        multi_class_tensor = super().__call__(image)
        return (multi_class_tensor, np.array(image))
