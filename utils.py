import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def save_gradcam(filename, gcam, raw_image=None, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = plt.cm.jet_r(gcam)[..., :3] * 255.0
    if raw_image is not None:
        raw_image = denormalize(raw_image).permute(1, 2, 0).cpu().numpy() * 255
        raw_image = raw_image[:, :, ::-1].copy()
        if paper_cmap:
            alpha = gcam[..., None]
            gcam = alpha * cmap + (1 - alpha) * raw_image
        else:
            gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    else:
        gcam = cmap
    cv2.imwrite(filename, np.uint8(gcam))

def gcam_to_mask(gcam, omega=100, sigma=0.5):
    mask = torch.sigmoid(omega * (gcam - sigma))
    return mask

def erase_mask(image, mask):
    return image * (1 - mask)

def tensor2imwrite(filename, tensor, from_pil_image=True):
    """ Converts a Tensor (based on Pillow image) to a OpenCV writable image. """
    img = tensor.detach().permute(1, 2, 0).cpu().numpy()
    if from_pil_image:
        img = (img * 255)[:, :, ::-1]
    cv2.imwrite(filename, img)

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def generate_segmentation_map(gcams, num_classes, img_shape, targets, threshold=0.5):
    """ Warning: this code doesn't handle batched tensors, only works for single items """
    combined_gcams = torch.zeros(num_classes + 1, *img_shape)
    combined_gcams[0] = 1e-5 #TODO: set this value to threshold and erase the other < threshold line
    gcams = F.interpolate(gcams, img_shape, mode='bilinear', align_corners=False)
    for gcam, target in zip(gcams, targets):
        gcam[gcam < threshold] = 0
        combined_gcams[target + 1, :, :] = gcam
    combined_gcams = combined_gcams.argmax(dim=0, keepdim=True)
    return combined_gcams

def bit_get(val, idx):
    """ Copied from https://github.com/tensorflow/models
    Gets the bit value.
    Args:
    val: Input value, int or numpy int array.
    idx: Which bit of the input val.
    Returns:
    The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """ Copied from https://github.com/tensorflow/models
    Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
    A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap    

def label_to_color_image(label):
    """ Copied (and edited) from https://github.com/tensorflow/models
    Adds color defined by the dataset colormap to the label.
    Args:
    label: A 2D array with integer type, storing the segmentation label.
    Returns:
    result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the dataset color map.
    Raises:
    ValueError: If label is not of rank 2
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label. Got {}'.format(label.shape))

    colormap = create_pascal_label_colormap()
    return colormap[label]

def save_segmentation_map(location, segmentation_map, input_image=None):
    segmentation_map = label_to_color_image(segmentation_map.squeeze())
    segmentation_map = cv2.cvtColor(segmentation_map.astype('uint8'), cv2.COLOR_RGB2BGR)
    if input_image is not None:
        input_image = input_image.detach().permute(1, 2, 0).cpu().numpy() * 255
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        segmentation_map = 0.3 * input_image.astype(np.float) + 0.7 * segmentation_map.astype(np.float)
    cv2.imwrite(location, segmentation_map)
