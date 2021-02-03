# xyz2image
Authors: Trond Henninen(trond.henninen@empa.ch) and Feng Wang

This script is for generating normalised Gaussian simulated annular dark-field scanning transmission electron microscopy (ADF-STEM) images from input atomic coordinates.
For rapidly generating a large dataset, it approximates a contrast similar to multislice simulated images by convolving a 2D Gaussian with the atomic coordinates. 
This is a decent approximation for up to 10 overlapping atoms, as the contrast is linearly additive for such thin lattices. 
Optimized for rapidly generating data with multiprocessing, so can generate millions of images per hour with a desktop processor. 

Inputs .xyz files and outputs a .tif image and compressed arrays (.npz) for fast save/load data e.g. for machine learning.
The input coodinates gets blurred by a 3D gaussian and has the z-dimension flattened to make the 2d image.  
xyz2image.xyz2image converts just one file, while xyz2image.folder convert all files in the folder.

Keep in mind, dimensions of the xyz coordinates, gauss_sigma and padxyz are all in ångström [å].

TODO: implement binary_2d and direct image output  
TODO: convert binary_radius to [å] instead if pixels  
TODO: set a parameter for voxsize (e.g. 9 pm), and calculate  

Variables:  
'folder' is the folder path where to look for .xyz files (default: '.' meaning current folder). 
'gauss_sigma' is the gaussian gauss_sigma in ångström [å]. (default: 0.4).  
'edgesize' is the size in pixels of the output square image and voxel cube (default: 128).  
'padxyz' is the minimum padding (in [å]) added around the atomic coordinate array, adjusted to make the box cubic (default: 1.5)  
'n_rotation' is the number of randomly rotated images (and 3D representations) that is generated for each xyz (1 means original orientation, while 8000 gives convenient files of 10s of MB)  
'n_stacks' number of .npz stacks to be generated for each input file  
'output_types' which type of outputs are generated (enable by setting the different options to True)   
'bitrate' bitrate of output files (should be 8 or 16)  
'binary_radius' radius of atoms in binary_3d/binary_2d (pixels)  

'frameshift' and 'maxframes' are for .xyz files with many frames:  
'frameshift' is which frame it start reading from (default 0)  
'maxframes' is how many consecutive frames is read (default 50)  
