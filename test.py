#!/usr/bin/env python

import sys
import itk
import vtk

if len(sys.argv) != 2:
    print('Usage: ' + sys.argv[0] + ' <InputFileName>')
    sys.exit(1)
imageFileName = sys.argv[1]

Dimension = 2
PixelType = itk.UC
ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(imageFileName)

itkToVtkFilter = itk.ImageToVTKImageFilter[ImageType].New()
itkToVtkFilter.SetInput(reader.GetOutput())

itkToVtkFilter.Update()
myvtkImageData = itkToVtkFilter.GetOutput()
print(myvtkImageData)