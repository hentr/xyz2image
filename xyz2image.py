'''
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
'n_stacks'  number of .npz stacks to be generated for each input file
'output_types' which type of outputs are generated (enable by setting the different options to True) 
'bitrate' bitrate of output files (should be 8 or 16)
'binary_radius' radius of atoms in binary_3d/binary_2d [pixels]

'frameshift' and 'maxframes' are for .xyz files with many frames:
'frameshift' is which frame it start reading from (default 0)
'maxframes' is how many consecutive frames is read (default 50)
'''

import os, glob
import numpy as np
from ase.io import read 
from scipy.ndimage.filters import gaussian_filter
from random import uniform
import tifffile as tif
import multiprocessing
from multiprocessing import Pool

########################   PARAMETERS   ###################################
class Parameters: #use an empty object to store parameters
    pass 
P = Parameters() # this object also becomes available to the subprocesses in the pool without being passed to xyz2image()
P.output_types = { #set True for which type of output you want
    'coordinates_3d':True, #3D coordinates of the rotated cluster
    'delta_3d':True, #binary 3D-array with 1 for atom center coordinates and 0 for rest
    'delta_2d':False, #binary 2D-array same as the 3D, but Z-coordinate has been collapsed
    'gaussian_3d':False, #delta_3d convolved with a 3D gaussian 
    'gaussian_2d':True, #Simulated image made by delta_2d convolved with a 2D gaussian
    'binary_3d':False, #binary 3D-array with 1 for spherical atoms with radius P.binary_radius, and 0 for rest
    'binary_2d':False, #NOT YET IMPLEMENTED binary 2D-array same as the 3D, but Z-coordinate has been collapsed
    'delta_2d_image':False, #delta_2d is also stored as a .tif stack 
    'gaussian_2d_image':False, #gaussian_2d is also stored as a .tif stack     
    'binary_2d_image':False, #NOT YET IMPLEMENTED binary_2d is also stored as a .tif stack
    }
P.folder = r'.'
P.output_folder = './npz_stacks' 
P.bitrate = 8 #output bitrate, 8 or 16
P.gauss_sigma = 0.4
P.edgesize = 80 
P.padxyz = 0.3
P.n_rotation = (1024)*8 #number of rotations in one .npz stack, 8k is a good compromise of speed/filesize/memory consumption
P.n_stacks = 1 # number of .npz stacks will be generated for each input file
P.frameshift = 0 
P.maxframes = 50
P.binary_radius = 7    #[pixels] radius of atoms in binary_3d
P.xyzfiles = glob.glob('*.xyz')

########################   /PARAMETERS   ###################################
     
def load(filename): 
    # for loading numpy compressed ND-arrays (.npz) files
    return(np.load(filename))

def xyz2image(file):
    print(file)
    fname = os.path.splitext(os.path.basename(file))[0] #gets the name of the file without the file extension
    t = read(file,index=':')
    t2 = t[min(P.frameshift,len(t)-1):min(P.frameshift+P.maxframes,len(t))] #frameshift and maxframes are for handling if multiple frames in the .xyz file
    print(file,len(t2))

    if not os.path.exists(P.output_folder): #make new folders if they don't exist
        os.makedirs(P.output_folder)
    if not os.path.exists(f'{P.output_folder}/{fname}'):
        os.makedirs(f'{P.output_folder}/{fname}')

    coordinates_3d_stack, delta_3d_stack, delta_2d_stack, gaussian_3d_stack = [],[],[],[]
    gaussian_2d_stack, binary_3d_stack, binary_2d_stack = [],[],[] 
    for at in t2: #for handling if multiple frames in the .xyz file 
        del at[at.numbers == 6]     #delete carbon atoms
        for rot in range(P.n_rotation):
            if rot == 0: # P.n_rotation == 1: #keep the first frame at same viewpoint as the input xyz file 
                at.rotate(90, 'z') 
            else: # random rotation
                at.euler_rotate(uniform(0,360),uniform(0,360),uniform(0,360)) 
            atoms = at.get_positions()
            atoms[:,0] -= min(atoms[:,0]); atoms[:,1] -= min(atoms[:,1]); atoms[:,2] -= min(atoms[:,2])
            maxx,maxy,maxz = max(atoms[:,0]),max(atoms[:,1]),max(atoms[:,2])
            maxxyz = max(maxx,maxy,maxz)
            padx,pady,padz = (maxxyz-maxx)/2+P.padxyz,(maxxyz-maxy)/2+P.padxyz, (maxxyz-maxz)/2+P.padxyz
            atoms[:,0] += padx; atoms[:,1] += pady; atoms[:,2] += padz;

            edgemax = maxxyz+2*P.padxyz
            voxsize = edgemax/(P.edgesize-1)
            sigpix = P.gauss_sigma/voxsize
            #print(file,' - ',len(atoms),at,', voxel size -',voxsize)

            normatoms = np.round(atoms/edgemax*(P.edgesize-1)) #normalize the coordinate box
            normatoms = normatoms.astype(int)
            delta_3d = np.zeros((P.edgesize,P.edgesize,P.edgesize))#,dtype=bool)
            delta_3d[normatoms[:,0],normatoms[:,1],normatoms[:,2]] = 1
            delta_2d = np.zeros((P.edgesize,P.edgesize))#,dtype=bool)
            delta_2d[normatoms[:,0],normatoms[:,1]] = 1
            
            if P.output_types['coordinates_3d'] == True:
                coordinates_3d_stack.append(normatoms)
            if P.output_types['delta_3d'] == True:
                delta_3d_stack.append(delta_3d)
            if P.output_types['delta_2d'] == True or P.output_types['delta_2d_image'] == True:
                delta_2d_stack.append(delta_2d)
            if P.output_types['gaussian_2d'] == True or P.output_types['gaussian_2d'] == True:
                gaussian_2d = gaussian_filter(delta_2d, sigpix)
                gaussian_2d /= np.max(gaussian_2d)/(2**P.bitrate-1)
                gaussian_2d = gaussian_2d.astype('uint'+str(P.bitrate))
                gaussian_2d_stack.append(gaussian_2d)
            if P.output_types['gaussian_3d'] == True:
                gaussian_3d = gaussian_filter(delta_3d, sigpix)
                gaussian_3d /= np.max(gaussian_3d)/(2**P.bitrate-1)
                gaussian_3d = gaussian_3d.astype('uint'+str(P.bitrate))
                gaussian_3d_stack.append(gaussian_3d)
            if P.output_types['binary_3d'] == True:
                binary_3d = np.zeros((P.edgesize,P.edgesize,P.edgesize),dtype=bool)
                for atom in normatoms:
                    y,x,z = np.ogrid[ -atom[0]:P.edgesize-atom[0], -atom[1]:P.edgesize-atom[1], -atom[2]:P.edgesize-atom[2] ]
                    mask = x*x + y*y + z*z <= P.binary_radius**2
                    binary_3d[mask] = True
                binary_3d_stack.append(binary_3d)
            # if P.output_types['binary_2d'] == True or P.output_types['binary_2d_image'] == True:

    output_stacks = {}
    if P.output_types['coordinates_3d'] == True:
        output_stacks['coordinates_3d'] =  np.asarray(coordinates_3d_stack)
    if P.output_types['delta_3d'] == True:
        output_stacks['delta_3d'] = np.asarray(delta_3d_stack).astype(bool)
    if P.output_types['delta_2d'] == True:
        output_stacks['delta_2d']= np.asarray(delta_2d_stack).astype(bool)
    if P.output_types['gaussian_2d'] == True:
        output_stacks['gaussian_2d'] = np.asarray(gaussian_2d_stack)
    if P.output_types['gaussian_3d'] == True:
        output_stacks['gaussian_3d'] = np.asarray(gaussian_3d_stack)
    if P.output_types['binary_3d'] == True:
        output_stacks['binary_3d'] = np.asarray(binary_3d_stack)

    simulated_files = len( glob.glob(f'{P.output_folder}/{fname}/*' ))
    file_name = f'{P.output_folder}/{fname}/{str(simulated_files+P.n_rotation*len(t2)).zfill(8)}' 
    
    if any([P.output_types['delta_3d'],P.output_types['delta_2d'],P.output_types['gaussian_2d'],P.output_types['gaussian_3d'],P.output_types['binary_3d']]):
        np.savez_compressed(file_name+'.npz',**output_stacks)    #these files can be loaded with np.load
    if P.output_types['delta_2d_image'] == True: 
        tif.imsave(file_name+'delta_2d.tif',np.invert(np.asarray(delta_2d_stack).astype(bool)))
    if P.output_types['gaussian_2d_image'] == True: 
        tif.imsave(file_name+'gaussian_2d.tif',np.asarray(gaussian_2d_stack))
    #if P.output_types['binary_2d_image'] == True: 
   

def folder_parallellized(P): #runs all the .xyz files in the folder, parallelized with one file per thread
    os.chdir(P.folder)
    threads = multiprocessing.cpu_count() 
    with Pool(threads) as p:
        p.map(xyz2image, P.xyzfiles) 

if __name__ == '__main__':
    for i in range(0,P.n_stacks):
        folder_parallellized(P)

    #to load .npz files: 
    #npz = (load(file_name+'.npz'))
    #print(npz.files,np.shape(npz['delta_3d']),np.shape(npz['gaussian_2d']))
