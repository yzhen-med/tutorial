""" pydicom处理dcm图像 """


import os
import os.path as osp
import pydicom
import numpy as np


def sort_dcm(slices, mode='InstanceNumber'):
    if mode == 'InstanceNumber':
        slices.sort(key=lambda x: float(x.InstanceNumber))
    elif mode == 'Position':
        slices.sort(key=lambda x: float(x.ImagePositionPatient[-1]))
    return slices


def load_dcm_with_hu(dcm_root):
    dcm_fns = [osp.join(dcm_root, f) for f in os.listdir(dcm_root) 
              if f.endswith(".dcm")]
    slices = [pydicom.dcmread(f) for f in dcm_fns]
    slices = sort_dcm(slices, mode='InstanceNumber')  # Z轴调整

    # 计算物理参数
    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = float(slices[0].SliceThickness)
    spacing = [pixel_spacing[0], pixel_spacing[1], slice_thickness]
    origin = [float(slices[0].ImagePositionPatient[i]) for i in range(3)]

    # 获取方向向量并判断翻转
    orientation = slices[0].ImageOrientationPatient
    row_vec = np.array(orientation[:3])  # X轴
    col_vec = np.array(orientation[3:])  # Y轴
    
    # 处理像素数据
    image = np.stack([s.pixel_array for s in slices])
    intercept = float(slices[0].RescaleIntercept)
    slope = float(slices[0].RescaleSlope)
    hu_image = image * slope + intercept
    if row_vec[0] < 0:  # X轴反向时水平翻转
        image = np.flip(image, axis=2)
    if col_vec[1] < 0:  # Y轴反向时垂直翻转
        image = np.flip(image, axis=1)

    metadata = {
        "spacing": spacing,
        "origin": origin,
        "seriesUID": slices[0].SeriesInstanceUID
    }
    return metadata, hu_image
