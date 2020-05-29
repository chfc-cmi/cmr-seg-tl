#!/usr/bin/env python
# coding: utf-8

import pydicom as dicom
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from pathlib import Path
from tqdm import tqdm
import re
import os
import math
from scipy import stats
from skimage.transform import rescale,rotate
from skimage.util import pad,crop

# inspired by https://stackoverflow.com/a/42320319
def get_largest_mode(elements):
    counts = {k:elements.count(k) for k in set(elements)}
    modes = sorted(dict(filter(lambda x: x[1] == max(counts.values()), counts.items())).keys())
    return modes[-1]

studies = glob("../raw/*/*/*/study")

def get_slice_info(study):
    images = glob(study+"/sax_*/*.dcm")
    slices = [x for x in images if re.search(r'IM-\d+-0001.*\.dcm$', x)]
    sliceinfo = dict()
    patientID = Path(slices[0]).parent.parent.parent.name
    # phases no longer tracked, fixed by copying/omitting frames in conversion, affects only one patient
    # phases = list()
    for s in slices:
        dcm = dicom.dcmread(s)
        seriesNumber = int(dcm.SeriesNumber)
        sliceLocation = float(dcm.SliceLocation)
        sliceOrientation = dcm.ImageOrientationPatient
        dim = (dcm.Rows, dcm.Columns)
        cardNumIm = int(dcm.CardiacNumberOfImages)
        pixelSpacing = (float(dcm.PixelSpacing[0]),float(dcm.PixelSpacing[1]))
        imagePosition = [float(x) for x in dcm.ImagePositionPatient]
        sliceSpacing = None
        if hasattr(dcm, 'SpacingBetweenSlices'):
            sliceSpacing = float(dcm.SpacingBetweenSlices)
        sliceinfo[s] = dict(
            file = s,
            dim = dim,
            location = sliceLocation,
            series=seriesNumber,
            phases=cardNumIm,
            sliceOrientation=sliceOrientation,
            pixelSpacing=pixelSpacing,
            sliceSpacing=sliceSpacing,
            imagePosition=imagePosition,
            patientID=patientID
        )
    return sliceinfo

# for angles in numpy see https://stackoverflow.com/a/13849249
def remove_inconsistent_planes(sliceinfo):
    modeOrientation = stats.mode([np.round(v["sliceOrientation"],3) for v in sliceinfo.values()])[0][0]
    modeNorm = np.cross(modeOrientation[:3],modeOrientation[3:])
    slicesToRemove = list()
    for k in sliceinfo.keys():
        sliceOrientation = sliceinfo[k]["sliceOrientation"]
        sliceinfo[k]["rotationRequired"] = False
        if not np.allclose(sliceOrientation,modeOrientation,atol=.01):
            thisNorm = np.cross(sliceOrientation[:3],sliceOrientation[3:])
            cosAngle = np.dot(modeNorm, thisNorm)/np.linalg.norm(modeNorm)/np.linalg.norm(thisNorm)
            isParallel = np.isclose(1, abs(cosAngle), atol=.001)
            if isParallel:
                sliceinfo[k]["rotationRequired"] = True
                # print("Plane is rotated but parallel (keep):", k) # no need to log here, will be logged if kept until final conversion
            else:
                print("{}\torientation\tslice removed\t{}\tangleBetweenNorms:{}".format(sliceinfo[k]["patientID"],k,np.degrees(np.arccos(np.clip(cosAngle,-1.0,1.0)))))
                slicesToRemove.append(k)
    for k in slicesToRemove:
        del sliceinfo[k]
    return modeOrientation

def remove_collissions(sliceinfo):
    sorted_slices = sorted(list(sliceinfo.keys()), key=lambda x: sliceinfo[x]['location'])
    patientID = sliceinfo[sorted_slices[0]]["patientID"]
    prevSlice = sorted_slices[0]
    prevSliceLoc = sliceinfo[prevSlice]['location']
    for s in sorted_slices[1:]:
        sliceLoc = sliceinfo[s]['location']
        if sliceLoc-prevSliceLoc < 1:
            toRemove = s
            if sliceinfo[prevSlice]['series'] == sliceinfo[s]['series']:
                if prevSlice < s:
                    toRemove = prevSlice
            else:
                if sliceinfo[prevSlice]['series'] < sliceinfo[s]['series']:
                    toRemove = prevSlice
            del sliceinfo[toRemove]
            if s == toRemove:
                print("{}\tcollission\tslice removed\t{}\t{} ({},{})".format(patientID,Path(toRemove),Path(prevSlice),sliceLoc,prevSliceLoc))
                continue
            else:
                print("{}\tcollission\tslice removed\t{}\t{} ({},{})".format(patientID,Path(toRemove),Path(s),prevSliceLoc,sliceLoc))
        prevSlice = s
        prevSliceLoc = sliceLoc

def get_slice_spacing_from_metadata(sliceinfo):
    spacing_from_metadata = None
    unique_metadata_spaces = np.unique([v['sliceSpacing'] for k,v in sliceinfo.items() if v['sliceSpacing'] is not None])
    if len(unique_metadata_spaces) > 1:
        k = list(sliceinfo.keys())[0]
        print("{}\tslice spacing: metadata\tnone\t\tobserved values: {}".format(sliceinfo[k]["patientID"],unique_metadata_spaces))
    elif len(unique_metadata_spaces == 1):
        spacing_from_metadata = unique_metadata_spaces[0]
    return spacing_from_metadata

def get_empirical_slice_spacing(sliceinfo):
    # mode of observed spacings
    sorted_slices = sorted(list(sliceinfo.keys()), key=lambda x: sliceinfo[x]['location'])
    slice_locations = [sliceinfo[x]['location'] for x in sorted_slices]
    empirical_spacing = stats.mode(np.round(np.diff(slice_locations),2))[0][0]
    return empirical_spacing

def add_missing_slices(sliceinfo, expected_spacing):
    sorted_slices = sorted(list(sliceinfo.keys()), key=lambda x: sliceinfo[x]['location'])
    slice_distances = np.diff([sliceinfo[x]['location'] for x in sorted_slices])
    double_dist = np.where(np.isclose(slice_distances,expected_spacing*2,atol=1))[0]
    for i in double_dist:
        sliceinfo[sorted_slices[i]+"-copy"] = sliceinfo[sorted_slices[i]].copy()
        sliceinfo[sorted_slices[i]+"-copy"]["location"] += expected_spacing
        print("{}\tslice spacing: missing\tprevious slice copied\t{}\texpected: {}, acutal: {}".format(sliceinfo[sorted_slices[i]]["patientID"],sorted_slices[i],expected_spacing,slice_distances[i]))

def remove_excessive_slices(sliceinfo, expected_spacing):
    sorted_slices = sorted(list(sliceinfo.keys()), key=lambda x: sliceinfo[x]['location'])
    slice_distances = np.diff([sliceinfo[x]['location'] for x in sorted_slices])
    proper_dist = np.isclose(slice_distances,expected_spacing,atol=1)
    inconsistent_spaces = np.where(proper_dist==False)[0]
    patientID = sliceinfo[sorted_slices[0]]["patientID"]
    for i in range(len(inconsistent_spaces)):
        spot_to_check = inconsistent_spaces[i]
        # check if outlier
        if slice_distances[spot_to_check] > 2.5*expected_spacing:
            to_remove = spot_to_check
            if spot_to_check > (len(sorted_slices)/2):
                to_remove = spot_to_check+1
            del sliceinfo[sorted_slices[to_remove]]
            print("{}\tslice spacing: outlier\tslice removed\t{}\texpected: {}, actual: {}".format(patientID, sorted_slices[to_remove], expected_spacing, slice_distances[spot_to_check]))
            return remove_excessive_slices(sliceinfo, expected_spacing)
        # check if interspersed additional slice
        elif spot_to_check<(len(slice_distances)-1) and abs(slice_distances[spot_to_check]+slice_distances[spot_to_check+1]-expected_spacing)<1:
            del sliceinfo[sorted_slices[spot_to_check+1]]
            print("{}\tslice spacing: interspersed\tslice removed\t{}\texpected: {}, actual: {} and {}".format(patientID, sorted_slices[spot_to_check+1], expected_spacing, slice_distances[spot_to_check], slice_distances[spot_to_check+1]))
            return remove_excessive_slices(sliceinfo, expected_spacing)
        # check if two interspersed additional slices
        elif spot_to_check<(len(slice_distances)-2) and abs(slice_distances[spot_to_check]+slice_distances[spot_to_check+1]+slice_distances[spot_to_check+2]-expected_spacing)<1:
            del sliceinfo[sorted_slices[spot_to_check+1]]
            del sliceinfo[sorted_slices[spot_to_check+2]]
            print("{}\tslice spacing: interspersed\tslice removed\t{}\texpected: {}, actual: {} and {}".format(patientID, sorted_slices[spot_to_check+1], expected_spacing, slice_distances[spot_to_check], slice_distances[spot_to_check+1]))
            print("{}\tslice spacing: interspersed\tslice removed\t{}\texpected: {}, actual: {} and {}".format(patientID, sorted_slices[spot_to_check+2], expected_spacing, slice_distances[spot_to_check+1], slice_distances[spot_to_check+2]))
            return remove_excessive_slices(sliceinfo, expected_spacing)
    # check top and bottom slice (keep if within 2.5mm - logged as "not fixed" in fix_slice_spacing)
    if proper_dist[1] and not proper_dist[0] and abs(slice_distances[0]-expected_spacing)>2.5:
        print("{}\tslice spacing: top\tslice removed\t{}\texpected: {}, actual: {}".format(patientID, sorted_slices[0], expected_spacing, slice_distances[0]))
        del sliceinfo[sorted_slices[0]]
        return remove_excessive_slices(sliceinfo, expected_spacing)
    if proper_dist[-2] and not proper_dist[-1] and abs(slice_distances[-1]-expected_spacing)>2.5:
        print("{}\tslice spacing: bottom\tslice removed\t{}\texpected: {}, actual: {}".format(patientID, sorted_slices[-1], expected_spacing, slice_distances[-1]))
        del sliceinfo[sorted_slices[-1]]
        return remove_excessive_slices(sliceinfo, expected_spacing)
    return

def fix_slice_spacing(sliceinfo):
    patientID = sliceinfo[list(sliceinfo.keys())[0]]["patientID"]
    # get metadata spacing only to report inconsistencies, always use empirical spacing (mode of observed spacings)
    spacing_from_metadata = get_slice_spacing_from_metadata(sliceinfo)
    empirical_spacing = get_empirical_slice_spacing(sliceinfo)
    if spacing_from_metadata and abs(empirical_spacing-spacing_from_metadata)>.1:
        print("{}\tslice spacing: empirical/metadata\tnone\t\t{} (empirical) vs {} (metadata)".format(patientID, empirical_spacing, spacing_from_metadata))
    add_missing_slices(sliceinfo, empirical_spacing)
    remove_excessive_slices(sliceinfo, empirical_spacing)
    sorted_slices = sorted(list(sliceinfo.keys()), key=lambda x: sliceinfo[x]['location'])
    slice_distances = np.diff([sliceinfo[x]['location'] for x in sorted_slices])
    proper_dist = np.isclose(slice_distances,empirical_spacing,atol=1)
    for i in np.where(proper_dist == False)[0]:
        print("{}\tslice spacing: special\tnone\t{}\texpected: {}, actual: {}, next slices: {}".format(patientID, sorted_slices[int(i)], empirical_spacing, slice_distances[i], sorted_slices[int(i)+1]))
    return empirical_spacing

def get_adjusted_pixel_array(pixel_array, target_size, scale_factors, rotation_angle):
    # http://nipy.org/nibabel/dicom/dicom_orientation.html#i-j-columns-rows-in-dicom
    # The dicom pixel_array has dimension (Y,X), i.e. X changing faster.
    # However, the nibabel data array has dimension (X,Y,Z,T), i.e. X changes the slowest.
    # We need to flip pixel_array so that the dimension becomes (X,Y), to be consistent
    # with nibabel's dimension.
    adjusted_pixel_array = pixel_array.transpose()
    adjusted_pixel_array = rescale(adjusted_pixel_array, scale_factors)
    adjusted_pixel_array = rotate(adjusted_pixel_array, rotation_angle)
    x_diff = adjusted_pixel_array.shape[0] - target_size[0]
    y_diff = adjusted_pixel_array.shape[1] - target_size[1]
    if x_diff > 0:
        to_crop = x_diff/2
        adjusted_pixel_array = crop(adjusted_pixel_array, ((math.ceil(to_crop),math.floor(to_crop)),(0,0)))
    elif x_diff < 0:
        to_pad = abs(x_diff/2)
        adjusted_pixel_array = pad(adjusted_pixel_array, ((math.ceil(to_pad),math.floor(to_pad)),(0,0)))
    if y_diff > 0:
        to_crop = y_diff/2
        adjusted_pixel_array = crop(adjusted_pixel_array, ((0,0),(math.ceil(to_crop),math.floor(to_crop))))
    elif y_diff < 0:
        to_pad = abs(y_diff/2)
        adjusted_pixel_array = pad(adjusted_pixel_array, ((0,0),(math.ceil(to_pad),math.floor(to_pad))))
    return adjusted_pixel_array

# heavily adjusted from https://github.com/baiwenjia/ukbb_cardiac/blob/master/data/biobank_utils.py by Wenjia Bai (under Apache-2 license)
def convert_dicom_stack_to_nii(sliceinfo, info, path=".",used_files_log=None,patientDim=None):
    """ Read dicom images and store them in a 3D-t volume. """
    sorted_slices = sorted(list(sliceinfo.keys()), key=lambda x: sliceinfo[x]['location'])
    
    # Number of slices
    Z = len(sorted_slices)
    
    # Read a dicom file at the first slice to get the temporal information
    X = info["dim"][1]
    Y = info["dim"][0]
    T = info["phases"]
    dx = info["pixelSpacing"][1]
    dy = info["pixelSpacing"][0]
    dz = info["sliceSpacing"]
    patientID = Path(sliceinfo[sorted_slices[0]]["file"]).parent.parent.parent.name
    globalScaleFactor = 1
    
    # if image is larger than 256 in any direction: scale down - ukbb_cardiac does not handle larger images properly (if structure is too large the fixed number of convolutions does not detect the correct features)
    if max(X,Y) > 256:
        globalScaleFactor = 256/max(X,Y)
        print("{}\timage size >256\tscaled\t\t({},{}) scaling down with factor: {}".format(patientID,X,Y,globalScaleFactor))
        orig_X = X
        orig_Y = Y
        orig_dx = dx
        orig_dy = dy
        X = round(X*globalScaleFactor)
        Y = round(Y*globalScaleFactor)
        dx = dx/globalScaleFactor
        dy = dy/globalScaleFactor
    
    if patientDim is not None:
        patientDim.append(dict(pid=patientID,X=X,Y=Y,pixelSpacingX=dx,pixelSpacingY=dy,sliceSpacing=dz,scaleFactor=globalScaleFactor))
    
    # DICOM coordinate (LPS)
    #  x: left
    #  y: posterior
    #  z: superior
    # Nifti coordinate (RAS)
    #  x: right
    #  y: anterior
    #  z: superior
    # Therefore, to transform between DICOM and Nifti, the x and y coordinates need to be negated.
    # Refer to
    # http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
    # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage

    # The coordinate of the upper-left voxel of the first and second slices
    pos_ul = np.array(sliceinfo[sorted_slices[0]]["imagePosition"])
    pos_ul[:2] = -pos_ul[:2]

    # Image orientation
    axis_x = np.array(info["modeOrientation"][:3])
    axis_y = np.array(info["modeOrientation"][3:])
    axis_x[:2] = -axis_x[:2]
    axis_y[:2] = -axis_y[:2]
    axis_z = np.cross(axis_x, axis_y)
    # check that all vectors are normalized
    axis_sizes = [np.linalg.norm(ax) for ax in [axis_x, axis_y, axis_z]]
    if not np.allclose(axis_sizes, 1, atol=.01):
        print("{}\taxis scaling\tnone\t\tlength of x,y,z (should all be 1): {}".format(patientID, axis_sizes))
    
    # Affine matrix which converts the voxel coordinate to world coordinate
    affine = np.eye(4)
    affine[:3, 0] = axis_x * dx
    affine[:3, 1] = axis_y * dy
    affine[:3, 2] = axis_z * dz
    affine[:3, 3] = pos_ul
    
    # The 4D volume
    volume = np.zeros((X, Y, Z, T), dtype='float32')

    # Go through each slice
    for z in range(0, Z):
        current_slice_info = sliceinfo[sorted_slices[z]]
        
        # determine transformations
        scale_factors = (globalScaleFactor,globalScaleFactor)
        if not np.allclose(info["pixelSpacing"], current_slice_info["pixelSpacing"]):
            print("{}\tpixel spacing\tscaling/padding/cropping\t{}\texpected: {}, actual: {}".format(patientID, current_slice_info["file"], info["pixelSpacing"], current_slice_info["pixelSpacing"]))
            scale_factors = (current_slice_info["pixelSpacing"][1]/dx,current_slice_info["pixelSpacing"][0]/dy)
        rotation_angle = 0
        if current_slice_info["rotationRequired"]:
            current_x = np.array(current_slice_info["sliceOrientation"][:3])
            current_x[:2] = -current_x[:2]
            rotation_cos = np.dot(axis_x, current_x)/np.linalg.norm(axis_x)/np.linalg.norm(current_x)
            rotation_angle = np.degrees(np.arccos(np.clip(rotation_cos,-1.0,1.0)))
            print("{}\trotation\trotation\t{}\t{} degrees".format(patientID,current_slice_info["file"],rotation_angle))
        if info["dim"][0] != current_slice_info["dim"][0] or info["dim"][1] != current_slice_info["dim"][1]:
            print("{}\timage size\tscaling/padding/cropping\t{}\texpected: {}, actual: {}".format(patientID, current_slice_info["file"], info["dim"], current_slice_info["dim"]))
        target_size = (X,Y)
        
        # We need to find the files for this slice
        files = glob(current_slice_info["file"].replace("-0001","-*",1))

        # sort the files according to the trigger time
        files_time = []
        for f in files:
            d = dicom.read_file(f)
            t = d.TriggerTime
            files_time += [[f, t]]
        files_time = sorted(files_time, key=lambda x: x[1])

        # Read the images
        for t in range(0, T):
            try:
                f = files_time[t][0]
                d = dicom.read_file(f)
                volume[:, :, z, t] = get_adjusted_pixel_array(d.pixel_array, target_size, scale_factors, rotation_angle)
            except IndexError:
                print('{}\tmissing frame\tprevious frame copied\t{}\tframe: {}'.format(patientID, f, t))
                volume[:, :, z, t] = volume[:, :, z, t - 1]
            if used_files_log is not None:
                used_files_log.write("{}\t{}\t{}\t{}\n".format(patientID,z,t,"\t".join(Path(f).parts[-2:])))

    # Temporal spacing
    dt = (files_time[1][1] - files_time[0][1]) * 1e-3

    # Store the image
    os.makedirs(path+"/"+patientID,exist_ok=True)
    nim = nib.Nifti1Image(volume, affine)
    nim.header['pixdim'][4] = dt
    nim.header['sform_code'] = 1
    # sizes are in [mm] and time is in [s] 
    nim.header['xyzt_units'] = 10
    nib.save(nim, path+"/"+patientID+"/sa.nii.gz")
    
    if volume.shape[0] > volume.shape[1]:
        # Store an additional rotated version if Rows<Columns - just for better segmentation performance of ukbb_cardiac (keeping affine is not correct but should not change results)
        os.makedirs(path+"/"+patientID+"_rot90",exist_ok=True)
        nim = nib.Nifti1Image(np.rot90(volume, 3), affine)
        nim.header['pixdim'][4] = dt
        nim.header['sform_code'] = 1
        nim.header['xyzt_units'] = 10
        nib.save(nim, path+"/"+patientID+"_rot90/sa.nii.gz")

f = open("../../../analysis/kaggle/used_dicoms.log", "w")
patientDim = list()

for i in tqdm(studies):
    sliceinfo = get_slice_info(i)
    modeOrientation = remove_inconsistent_planes(sliceinfo)
    remove_collissions(sliceinfo)
    sliceSpacing = fix_slice_spacing(sliceinfo)
    pixelSpacing = get_largest_mode([v["pixelSpacing"] for v in sliceinfo.values()])
    dim = get_largest_mode([v["dim"] for v in sliceinfo.values()])
    phases = get_largest_mode([v["phases"] for v in sliceinfo.values()])
    info = dict(modeOrientation=modeOrientation, pixelSpacing=pixelSpacing, dim=dim, phases=phases, sliceSpacing=sliceSpacing)
    convert_dicom_stack_to_nii(sliceinfo, info, path=".", used_files_log=f, patientDim=patientDim)

f.close()

pd.DataFrame(patientDim).to_csv("../../../analysis/kaggle/patient_dimensions.tsv", sep="\t", index=False)
