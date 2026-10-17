"""
@author: Tina Hsu
Functions for segmentation using Mesmer.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import zoom
from skimage.exposure import rescale_intensity
import skimage.io as skio
from skimage.transform import resize as skimage_resize
from skimage.measure import regionprops_table
from deepcell.applications import Mesmer
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay


def run_mesmer(nuclear_channel, cytoplasm_channels, in_dir, out_dir, deepcell_token,
               mes_image_mpp, mes_interior_threshold=0.2, mes_maxima_threshold=0.075,
               kernel_size=3, norm_method="minmax", resize=False,
               plot_mask=False, plot_overlay=False, output_pred_df=False):
    """
    Running Mesmer.
    Note: panel.csv is required with a column called 'Target'.
    :param nuclear_channel: Number of the channel for nuclear, type: list[int]
    :param cytoplasm_channels: Number of the channels for cytoplasm, type: list[int]
    :param in_dir: Directory with tiff files, type: Path
    :param out_dir: Directory where results will be saved, type: Path
    :param deepcell_token: Deepcell user token, type: str
    :param mes_image_mpp: Microns per pixel for images under path_dir, type: float
    :param mes_interior_threshold: Control how conservative the model is in estimating a cell vs background.
    Lower values will result in larger cells; higher values will result in smaller cells, set to default, type: float
    :param mes_maxima_threshold: Control what the model considers a unique cell. Lower values will result in more
    separate cells being predicted; higher values will result in fewer cells, set to default, type: float
    :param kernel_size: Kernel size for Mesmer, set to default, type: int
    :param norm_method: Normalization method for pixel intensity of different channels, set to default, type: str
    :param resize: Resizing the image or not, set to default, type: bool
    :param plot_mask: Output the plots of mask from Mesmer, type: bool
    :param plot_overlay: Output the overlay plots of segmentation from Mesmer, type:bool
    :param output_pred_df: Store the prediction dataframe of segmentation from Mesmer, type: bool
    :return:
    """
    # Update token
    os.environ.update({"DEEPCELL_ACCESS_TOKEN": deepcell_token})

    # Declare instance
    app = Mesmer()

    # Iteratively quantify expression
    exp_dfs = []

    for roi, image_path in enumerate(list(in_dir.rglob("*.tif")) + list(in_dir.rglob("*.tiff")), start=1):
        file_name = os.path.basename(image_path)
        print(f"Processing {file_name}")

        # Read image
        img = skio.imread(image_path)

        # Remove noise
        nuclei, cytoplasm = image_preprocess(img, nuclear_channel, cytoplasm_channels, kernel_size, norm_method)

        x = stack_for_mesmer(nuclei, cytoplasm)

        # Compress if resize
        if resize:
            original_shape = nuclei.shape
            x = __resize_image(x, target_size=(512, 512))

        # Segment
        seg_pred = app.predict(x,
                               image_mpp=mes_image_mpp,
                               postprocess_kwargs_whole_cell={
                                   'maxima_threshold': mes_maxima_threshold,
                                   'interior_threshold': mes_interior_threshold}
                               )

        # Expand if resized
        if resize:
            seg_pred = __resize_image(x, target_size=original_shape)
        # (1, 1165, 1226, 1)
        pred_final = __squeeze_image(seg_pred)

        # Save mask
        if plot_mask:
            skio.imsave(out_dir / f"{file_name}_mask.tiff", pred_final.astype(np.float32), check_contrast=False)

        # Plot final segment
        if plot_overlay:
            __plot_seg_pred(x, seg_pred, out_dir,
                            name=f"interior{mes_interior_threshold}_maxima{mes_maxima_threshold}_{file_name}")

        # Output the prediction only to csv
        if output_pred_df:
            pd.DataFrame(pred_final).to_csv(out_dir / f"{file_name}_mesmer_pred.csv", index=False)

        # Record expression
        panel = load_panel(in_dir)
        exp_df = __quantify_expression(img, pred_final, panel, file_name, roi)
        exp_dfs.append(exp_df)

    exp_dfs = pd.concat(exp_dfs, ignore_index=True)
    exp_dfs.to_parquet(out_dir / "measurements.parquet", index=False)


def load_panel(in_dir):
    """
    Get the marker and its corresponding channel/index in the panel.
    :param in_dir: Directory with tiff files, type: str
    :return: Marker name to its channel in panel, type: dict[str, int]
    """
    panel_df = pd.read_csv(in_dir / "panel.csv")

    return {name: int(index) for name, index in zip(panel_df['Target'], panel_df.index)}


def image_preprocess(img, nuclear_channel, cytoplasm_channels, kernel_size, norm_method):
    """
    Prepare input for Mesmer.
    :param img: Pixel intensity values of the image, type: np.array
    :param nuclear_channel: Corresponding number of the channel for nuclear, type: int
    :param cytoplasm_channels: List of the numbers of channels for cytoplasm, type: tuple
    :param kernel_size: Window size for median filtering, type: int
    :param norm_method: Normalization method, type: str
    :return: Pixel intensity values of the image with all the channels, type: np.array
    """
    nuclei = remove_salt_pepper_noise_normalize(img, kernel_size, norm_method, nuclear_channel)
    cytoplasm = remove_salt_pepper_noise_normalize(img, kernel_size, norm_method, cytoplasm_channels)

    return nuclei, cytoplasm


def remove_salt_pepper_noise_normalize(img, kernel_size, norm_method, channels):
    """
    Remove salt and pepper noise and do normalization.
    :param img: Pixel intensity values of the image, type: np.array
    :param kernel_size: Window size for median filtering, type: int
    :param norm_method: Normalization method, type: str
    :param channels: List of the channels, type: list[str]
    :return: Sum of pixel intensity of the image with all channels, type: np.array
    """

    return sum(
        normalize(median_filter(img[i, :, :], kernel_size=kernel_size), method=norm_method) for i in channels)


def median_filter(img_channel, kernel_size):
    """
    Apply a median filter to the image.
    :param img_channel: Pixel intensity values of the image for one specific channel, type: np.array
    :param kernel_size: Window size for median filtering, type: int
    :return: Median filtered pixel values of the image, type: np.array
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd integer")

    return cv2.medianBlur(img_channel.astype(np.float32), kernel_size)


def normalize(image, method):
    """
    Pixel normalization.
    :param image: Median filtered pixel values of the image, type: np.array
    :param method: Normalization method, type: str
    :return Pixel intensity of the image after normalization, type: np.array
    """
    if method == 'minmax':
        min_val, max_val = np.min(image), np.max(image)
        if np.isclose(min_val, max_val):
            return np.zeros_like(image)

        return (image - min_val) / (max_val - min_val)

    elif method == 'percentile':
        low, high = np.percentile(image, (1, 99))
        if np.isclose(low, high):
            return np.zeros_like(image)

        return rescale_intensity(image, in_range=(low, high), out_range=(0, 1))


def stack_for_mesmer(nuclei, cytoplasm):
    """
    Merge and create stack for running Mesmer.
    :param nuclei: Pixel intensity of image with nuclei channel, type: np.array
    :param cytoplasm: Pixel intensity of image with all cytoplasm channels, type: np.array
    :return: Pixel intensity of image with nuclei and cytoplasm channels in 4 dimensions, type: np.array
    """

    return np.expand_dims(np.stack([nuclei, cytoplasm], axis=-1), axis=0)


def __resize_image(image, target_size, method='bicubic'):
    """
    Resize the image to target size.
    :param image: Pixel intensity of image with nuclei and cytoplasm, type: np.array
    :param target_size: Target size of the image, type: tuple
    :param method: Method used to resize image in ('nearest', 'bilinear', 'bicubic'), type: str
    :return: Pixel intensity of the image in target size, type: np.array
    """
    if image.shape[0] != 1 or image.ndim != 4 or image.shape[3] not in [1, 2]:
        raise ValueError("Image must have shape [1, x, y, 2] or [1, x, y, 1]")

    current_height, current_width = image.shape[1:3]
    target_height, target_width = target_size
    channels = image.shape[3]

    # Determine if we're upscaling or downscaling
    is_upscaling = target_height > current_height or target_width > current_width

    # For upscaling, always use 'nearest' to avoid interpolation
    if is_upscaling:
        method = 'nearest'

    if method == 'nearest':
        order = 0
    elif method == 'bilinear':
        order = 1
    elif method == 'bicubic':
        order = 3
    else:
        raise ValueError("Unsupported method. Use 'nearest', 'bilinear', or 'bicubic'.")

    if method in ['nearest', 'bilinear']:
        zoom_factors = (1, target_height / current_height, target_width / current_width, 1)
        resized = zoom(image, zoom_factors, order=order)
    else:
        resized = np.zeros((1, target_height, target_width, channels), dtype=image.dtype)
        for i in range(channels):
            resized[0, :, :, i] = skimage_resize(image[0, :, :, i], (target_height, target_width),
                                                 order=3, mode='reflect', anti_aliasing=True)

    return resized


def __squeeze_image(image):
    """
    Remove single-dimensional entries that are not necessary for the image.
    :param image: Pixel intensity of the image, type: np.array
    :return: Pixel intensity of image after squeezing, type: np.array
    """
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[3] != 1:
        raise ValueError("Input image must have shape (1, x, y, 1)")

    reshaped_image = image.squeeze()

    return reshaped_image


def __quantify_expression(imc_image, mask, panel_info, image_name, roi_number):
    """
    Process a single IMC image and its mask, creating ROI-based unique object IDs.
    :param imc_image: Pixel intensity of the original image, type: np.array
    :param mask: Pixel intensity of the segmented image, type: np.array
    :param panel_info: Channels and their corresponding indices, type: Dict[str, int]
    :param image_name: Image ID/name, type: str
    :param roi_number: The number/order of the ROI in the image list, type: int
    :return: Intensity measurement of the image, type: pd.DataFrame
    """
    mask = __ensure_valid_mask(mask)
    geometric_props = __calculate_geometric_properties(mask)
    df = __create_base_dataframe(geometric_props, image_name, roi_number)

    return __add_intensity_measurements(df, mask, imc_image, panel_info)


def __ensure_valid_mask(mask):
    """
    Ensure the mask is valid for regionprops analysis without unnecessary relabeling.
    :param mask: Pixel intensity of the segmented image, type: np.array
    :return: Valid mask for analysis, type: np.array
    """
    if not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(int)

    unique_labels = np.unique(mask)
    if 0 in unique_labels:
        unique_labels = unique_labels[unique_labels != 0]

    if np.array_equal(unique_labels, np.arange(1, len(unique_labels) + 1)):

        return mask
    else:
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}
        label_map[0] = 0

        return np.vectorize(label_map.get)(mask)


def __calculate_geometric_properties(mask):
    """
    Calculate the geometric info of the segmentation result.
    :param mask: Valid mask, type: np.array
    :return: Geometric properties of the cells after segmentation, type: Dict[str, np.array]
    """
    props = regionprops_table(
        mask,
        properties=(
            'label', 'centroid', 'area', 'perimeter',
            'eccentricity', 'major_axis_length', 'minor_axis_length'
        )
    )
    props['circularity'] = 4 * np.pi * props['area'] / (props['perimeter'] ** 2)

    return props


def __create_base_dataframe(props, image_name, roi_number):
    """
    Create a DataFrame with geometric properties and ROI-based unique object IDs.
    :param props: Geometric properties of the cells after segmentation, type: Dict[str, np.ndarray]
    :param image_name: Image name, type: str
    :param roi_number: Number of ROI in the image list, type: int
    :return: Geometric and meta info of the cells, type: pd.DataFrame
    """
    df = pd.DataFrame({
        'Image': image_name,
        'ROI': roi_number,
        'Centroid X px': props['centroid-1'],
        'Centroid Y px': props['centroid-0'],
        'Cell: Area': props['area'],
        'Cell: Perimeter': props['perimeter'],
        'Cell: Circularity': props['circularity'],
        'Cell: Max caliper': props['major_axis_length'],
        'Cell: Min caliper': props['minor_axis_length'],
        'Cell: Eccentricity': props['eccentricity'],
    })

    df['Object ID'] = df.apply(lambda row: f"roi{roi_number}_{props['label'][row.name]}", axis=1)

    columns = ['Image', 'Object ID', 'ROI'] + [col for col in df.columns if col not in ['Image', 'Object ID', 'ROI']]
    df = df[columns]

    return df


def __add_intensity_measurements(df, mask, imc_image, panel_info):
    """
    Add intensity measurements for each marker to the base DataFrame.
    :param df: Geometric and meta info of the cells, type: pd.DataFrame
    :param mask: Valid mask for analysis, type: np.array
    :param imc_image: Pixel intensity of the original image, type: np.array
    :param panel_info: Channels and their corresponding indices, type: Dict[str, int]
    :return: Segmentation result with geometric info and intensity of markers, type: pd.DataFrame
    """
    intensity_data = {}
    for marker, channel_index in panel_info.items():
        channel_data = imc_image[channel_index]

        intensity_props = __calculate_intensity_properties(mask, channel_data)
        for prop, col_name in zip(
                ['intensity_mean', 'intensity_std', 'intensity_max', 'intensity_min'],
                ['mean', 'std dev', 'max', 'min']
        ):
            intensity_data[f'Cell: {marker} {col_name}'] = intensity_props[prop]

    intensity_df = pd.DataFrame(intensity_data)

    return pd.concat([df, intensity_df], axis=1)


def __calculate_intensity_properties(mask, channel_data):
    """
    Calculate intensity properties for a single channel.
    :param mask: Valid mask for analysis, type: np.array
    :param channel_data: Pixel intensity of image for single channel, type: np.array
    :return: Intensity info of a single channel, type: Dict[str, np.array]
    """

    return regionprops_table(
        mask,
        intensity_image=channel_data,
        properties=('label', 'intensity_mean', 'intensity_std', 'intensity_max', 'intensity_min')
    )


def __plot_seg_pred(x, seg_pred, out_dir, name):
    """
    Plot the segmentation result from Mesmer.
    :param x: Pixel intensity of the image in 4-dimension, type: np.array
    :param seg_pred: Prediction result of segmentation from Mesmer, type: np.array
    :param out_dir: Directory to store the figure, type: str
    :return:
    """
    rgb_images = create_rgb_image(x, channel_colors=['green', 'blue'])
    overlay_data = make_outline_overlay(rgb_data=rgb_images, predictions=seg_pred)

    plt.figure(figsize=(7, 7))

    plt.imshow(overlay_data[0])

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(out_dir / f"{name}.png")
    plt.clf()
