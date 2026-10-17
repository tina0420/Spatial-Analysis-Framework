"""
@author: Tina Hsu
Functions for extracting MCD files for IMC studies.
"""
import os
import re
from pathlib import Path
import pandas as pd
from readimc import MCDFile
import xtiff
from xml.etree import ElementTree as ET


def extract_mcd_for_imc(in_dir, out_dir):
    """
    Extract the panel and tiff files in MCD
    :param in_dir: Directory with tiff files, type: Path
    :param out_dir: Directory where results will be saved, type: Path
    :return:
    """
    # Create output directory
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Load all paths of files in directory
    file_list = list(in_dir.rglob("*.mcd"))

    # Iteratively extract tiff files and its panel
    for mcd_path in file_list:
        file_name = os.path.basename(mcd_path)
        print(f"Processing {file_name}")

        __extract_panel_from_mcd(mcd_path, out_dir)
        __extract_tiffs_from_mcd(mcd_path, out_dir)


def __extract_panel_from_mcd(mcd_file, out_dir):
    """
    Extracting the panel from MCD metadata.
    :param mcd_file: MCD file path, type: Path
    :param out_dir: output path, type: Path
    :return:
    """
    with MCDFile(mcd_file) as MCD_handle:
        panel_df = pd.DataFrame(
            [MCD_handle.slides[0].acquisitions[0].channel_names, MCD_handle.slides[0].acquisitions[0].channel_labels]).T
        panel_df.columns = ["Metal Tag", "Target"]
        panel_df.to_csv(out_dir / "panel.csv", index=False)


def __extract_tiffs_from_mcd(mcd_file, out_dir):
    """
    Extracting tiff files from MCD.
    :param mcd_file: MCD file path, type: Path
    :param out_dir: output path, type: Path
    :return:
    """
    with MCDFile(mcd_file) as MCD_handle:
        for slide in MCD_handle.slides:
            for acquisition in slide.acquisitions:
                try:
                    xtiff.to_tiff(img=MCD_handle.read_acquisition(acquisition),
                                  file=Path(out_dir) / f"{Path(mcd_file).stem}_s{slide.id}_a{acquisition.id}.ome.tiff",
                                  channel_names=acquisition.channel_labels,
                                  ome_xml_fun=__get_acquisition_ome_xml,
                                  channel_fluors=acquisition.channel_names,
                                  xml_metadata=MCD_handle.metadata.replace("\r\n", ""))
                except Exception as e:
                    raise Exception(f"Error reading acquisition {acquisition.id} from file {MCD_handle.path.name}: {e}")


def __get_acquisition_ome_xml(img, image_name, channel_names, big_endian, pixel_size, pixel_depth,
                              channel_fluors=None, xml_metadata=None, **ome_xml_kwargs):
    element_tree = xtiff.get_ome_xml(img, image_name, channel_names, big_endian, pixel_size, pixel_depth,
                                     **ome_xml_kwargs)

    root_elem = element_tree.getroot()
    root_elem.set("Root", "Sakura")

    if channel_fluors is not None:
        assert len(channel_fluors) == img.shape[2]
        channel_elems = element_tree.findall("./Image/Pixels/Channel")
        assert channel_elems is not None and len(channel_elems) == img.shape[2]
        for channel_elem, channel_fluor in zip(channel_elems, channel_fluors):
            channel_elem.set("Fluor", channel_fluor)

    if xml_metadata is not None:
        structured_annot_elem = ET.SubElement(root_elem, "StructuredAnnotations")
        xml_annot_elem = ET.SubElement(structured_annot_elem, "XMLAnnotation")
        xml_annot_elem.set("ID", "Annotation:0")
        xml_annot_value_elem = ET.SubElement(xml_annot_elem, "Value")
        orig_metadata_elem = ET.SubElement(xml_annot_value_elem, "OriginalMetadata")
        orig_metadata_key_elem = ET.SubElement(orig_metadata_elem, "Key")
        orig_metadata_key_elem.text = "MCD-XML"
        orig_metadata_value_elem = ET.SubElement(orig_metadata_elem, "Value")
        orig_metadata_value_elem.text = xml_metadata

    return element_tree


def quantify_from_qupath(path_dir):
    """
    Process QuPath measurements.
    :param path_dir: input path
    :return: [1] expression matrix, [2] location matrix, type: pd.DataFrame
    """
    # Read dataframes
    df = pd.read_csv(path_dir / "measurements.csv")

    channels = ['Image', 'Cell: Area']

    for col in df.columns:
        if re.search('Centroid', col):
            channels.append(col)

    for col in df.columns:
        if re.search('Cell', col):
            if re.search('mean', col):
                channels.append(col)
    df = df.loc[:, channels]

    # Subset expression columns and normalize
    exp_df = df.iloc[:, 4:]
    exp_df = pd.concat([df.iloc[:, 0], exp_df], axis=1)

    # Subset location columns
    loc_df = df.loc[:, channels[:4]]
    loc_df.columns = ["Image", "Area", "X", "Y"]

    return exp_df, loc_df


def quantify_from_cellprofiler(path_dir, set_count=None):
    """
    Process CellProfiler outputs.
    If set_count is a number then we run multiple files, else set as None

    :param path_dir: input path
    :param set_count: split to individual files when size is too big, input the number of ROIs, type: int
    :return: [1] expression matrix, [2] location matrix, [3] relation matrix, type: pd.DataFrame
    """
    # Read dataframes
    if set_count:
        cell_arr = []
        rel_arr = []
        for i in range(1, set_count + 1):
            cell_df = pd.read_csv(path_dir + f"/cell_{i}.csv")
            rel_df = pd.read_csv(path_dir + f"/Object relationships_{i}.csv")
            cell_arr.append(cell_df)
            rel_arr.append(rel_df)
        cell_df = pd.concat(cell_arr).reset_index(drop=True)
        rel_df = pd.concat(rel_arr).reset_index(drop=True)
    else:
        cell_df = pd.read_csv(path_dir + "/cell.csv")
        rel_df = pd.read_csv(path_dir + "/Object relationships.csv")
    panel_df = pd.read_csv(path_dir + "/panel.csv")

    # Subset expression columns
    exp_cols = [i for i in cell_df.columns if i.startswith("Intensity_MeanIntensity_FullStack_")]
    exp_df = cell_df[exp_cols]

    # Renaming according to panel
    panel_arr = panel_df[panel_df.full == 1].Target.values
    cols_dict = dict(zip(["c" + str(i) for i in range(1, len(panel_arr) + 1)], panel_arr))
    exp_df.columns = [cols_dict[i] for i in [i.split("_")[-1] for i in exp_df.columns]]

    # Scaling up and normalize
    exp_df = exp_df * 1E5
    exp_df = pd.concat([cell_df.iloc[:, 0], exp_df], axis=1)

    # Subset location columns
    loc_cols = ['ImageNumber', 'AreaShape_Area', 'AreaShape_Center_X', 'AreaShape_Center_Y']
    loc_df = cell_df[loc_cols]
    loc_df.columns = ["ImageNumber", "Area", "X", "Y"]

    # Subset relation columns
    rel_cols = ['First Image Number', 'First Object Number', 'Second Object Number']
    rel_df = rel_df[rel_cols]
    rel_df.columns = ["ImageNumber", "Cell", "Neighbour"]

    return exp_df, loc_df, rel_df
