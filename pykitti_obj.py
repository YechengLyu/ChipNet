"""Provides 'obj', which loads and parses KITTI detection data."""

import datetime as dt
import glob
import os
from collections import namedtuple
from PIL import Image

import numpy as np

import pykitti.utils as utils

__author__ = "Yecheng Lyu"
__email__ = "ylyu@wpi.edu"


class obj:
    """Load and parse objection detection data into a usable format."""

    def __init__(self, base_path, **kwargs):
        self.data_path = base_path
        self.frames = kwargs.get('frames', None)
        # Setting imformat='cv2' will convert the images to uint8 and BGR for
        # easy use with OpenCV.
        self.imformat = kwargs.get('imformat', None)

        # Default image file extension is '.png'
        self.imtype = kwargs.get('imtype', 'png')
        
        
    def __len__(self):
        """Return the number of frames loaded."""
        if (self.frames==None):
            velo_path = os.path.join(self.data_path, 'velodyne','*.bin')
            velo_files = sorted(glob.glob(velo_path))
            return len(velo_files)
        else:
            return len(self.frames)
    
    
    @property
    def name(self):
        """Generator to read image files for cam2 (RGB left)."""
        impath = os.path.join(self.data_path, 'image_2','*.{}'.format(self.imtype))
        imfiles = sorted(glob.glob(impath))
        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            imfiles = [imfiles[i] for i in self.frames]
        
        for filename in imfiles:
            words=filename.replace('/',' ').split()
            res_name=words[-1]
            yield res_name



    @property
    def cam2(self):
        """Generator to read image files for cam2 (RGB left)."""
        impath = os.path.join(self.data_path, 'image_2','*.{}'.format(self.imtype))
        imfiles = sorted(glob.glob(impath))
        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            imfiles = [imfiles[i] for i in self.frames]

        # Return a generator yielding the images
#         print imfiles[0:5]
        return utils.get_images(imfiles, self.imformat)


    @property
    def velo(self):
        """Generator to read velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files
        velo_path = os.path.join(self.data_path, 'velodyne','*.bin')
        velo_files = sorted(glob.glob(velo_path))

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            velo_files = [velo_files[i] for i in self.frames]

        # Return a generator yielding Velodyne scans.
        # Each scan is a Nx4 array of [x,y,z,reflectance]
#         print velo_files[0:5]
        return utils.get_velo_scans(velo_files)
    
    @property
    def velo_ford(self):
        """Generator to read velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files
        velo_path = os.path.join(self.data_path, 'velodyne','*.bin')
        velo_files = sorted(glob.glob(velo_path))

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            velo_files = [velo_files[i] for i in self.frames]

        # Return a generator yielding Velodyne scans.
        # Each scan is a Nx4 array of [x,y,z,reflectance]
#         print velo_files[0:5]
        for filename in velo_files:
            scan = np.fromfile(filename, dtype=np.float32)
            yield scan.reshape((-1, 5))


    @property
    def calib(self):
        """Read a rigid transform calibration file as a numpy.array."""
        # Find all the Calibration files
        calib_path = os.path.join(self.data_path, 'calib','*.txt')
        calib_files = sorted(glob.glob(calib_path))
        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            calib_files = [calib_files[i] for i in self.frames]

        for file in calib_files:
            yield file
        
        
    @property
    def gt(self):
        """Read a rigid transform calibration file as a numpy.array."""
        # Find all the Calibration files
        gt_path = os.path.join(self.data_path, 'label_2','*.txt')
        gt_files = sorted(glob.glob(gt_path))
        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            gt_files = [gt_files[i] for i in self.frames]

        for file in gt_files:
            yield file
    
    @property
    def gt_img(self):
        """Read a rigid transform calibration file as a numpy.array."""
        # Find all the Calibration files
        gt_path = os.path.join(self.data_path, 'gt_image_2','*_road_*.png')
        gt_files = sorted(glob.glob(gt_path))
#         print gt_files
        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            gt_files = [gt_files[i] for i in self.frames]

        for file in gt_files:
            yield file
            
    @property        
    def gt_img_ford(self):
        """Read a rigid transform calibration file as a numpy.array."""
        # Find all the Calibration files
        gt_path = os.path.join(self.data_path, 'gt_image_2','*.png')
        gt_files = sorted(glob.glob(gt_path))
#         print gt_files
        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            gt_files = [gt_files[i] for i in self.frames]

        for file in gt_files:
            yield file       
            
    @property
    def gt_road_BEV(self):
        """Read road segmentation ground truth on top view"""
        # Find all the Calibration files
        gt_path = os.path.join(self.data_path, 'gt_image_BEV','*_road_*.png')
        gt_files = sorted(glob.glob(gt_path))
        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            gt_files = [gt_files[i] for i in self.frames]
            
#         print gt_files[0:5]

        for file in gt_files:
            yield file

            
            
            
            
def read_gt_file(filepath):
    data=[]
    category=[]
    value=[]
    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                items = line.split()
                data_line=items[0:1]
                value=[float(x) for x in items[1:-1]]
                data_line.extend(value)
                data.append(data_line)
            except ValueError:
                pass
    

    return data
def read_gt_img(filepath):
    gt      =Image.open(filepath)
    gt_arr  = np.uint8(gt)
    gt_arr  = gt_arr[:,:,2]/255
    return gt_arr


def read_gt_BEV(filepath):
    gt      =Image.open(filepath)
    gt_arr  = np.uint8(gt)
    gt_arr  = gt_arr[:,:,2]/255
    return gt_arr
    
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':')
#                 print key
#                 print value
            # The only non-float values in these files are dates, which
            # we don't care about anyway
#             try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass


    Tr_velo_to_cam=data['Tr_velo_to_cam'].reshape(3,4)
    Tr_velo_to_cam=np.vstack((Tr_velo_to_cam,[0,0,0,1]))
    data['Tr_velo_to_cam']=Tr_velo_to_cam

    R0_rect=data['R0_rect'].reshape(3,3)
    T0_rect=np.vstack((np.hstack((R0_rect,(np.array([0,0,0])).reshape(3,1))),[0,0,0,1]))
    data['T0_rect']=T0_rect



    P2=data['P2'].reshape(3,4)
    P2=np.vstack((P2,[0,0,0,1]))
    data['P2']=P2


    Tr_velo_to_img=(P2.dot(T0_rect)).dot(Tr_velo_to_cam)
    Tr_velo_to_img=Tr_velo_to_img[0:3,:]
#     print Tr_velo_to_img


    return np.float32(Tr_velo_to_img)

def Trans_velo_to_img(Tr_velo_to_img,velo):
    velo=np.transpose(velo)
    velo[3][:]=1
    x=Tr_velo_to_img.dot(velo)
    x[0][:]=x[0][:]/x[2]
    x[1][:]=x[1][:]/x[2]
    
    return np.transpose(x)
    
    
def xyz2sph(xyz):  # x,y,z,intensity,rho,polar,azimuthal
    ptsnew = np.hstack((xyz, np.zeros((xyz.shape[0],3))))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,-3] = np.sqrt(xy + xyz[:,2]**2)         # radial distance r
    ptsnew[:,-2] = np.arctan2(xyz[:,2], np.sqrt(xy)) # polar angle phi
    
#     theta=np.arctan2(xyz[:,1], xyz[:,0])    # azimuthal angle theta
#     theta[theta<0]+=2*np.pi
#     ptsnew[:,-1] = theta
    ptsnew[:,-1] = np.arctan2(xyz[:,1], xyz[:,0])    # azimuthal angle theta
    
    return ptsnew




def sph_prepare(sph,half_FOV=45):
#     half_FOV=45
    data=np.zeros([64,half_FOV*4,14])
    theta = 0
    line  = 0
    shift_value = 0.5*np.pi/180
    i_shift =0
    left=theta
    right=theta+shift_value
    vec=np.zeros([0,7])
    for i_pt in range(len(sph)):
        

        if (-half_FOV*np.pi/180<sph[i_pt][6] and sph[i_pt][6]<half_FOV*np.pi/180):
            if (left<sph[i_pt][6] and sph[i_pt][6]<right):
                vec=np.vstack((vec,sph[i_pt]))
            else:
                if vec.shape[0]>0:
                    vec=vec[np.argsort(vec[:,4]),:]
                    data[line,-(i_shift+half_FOV*2),:]=np.hstack((vec[0],vec[-1]))
                else:
                    pass



                if (i_shift==-1):
    #                 print (i_pt)
                    line+=1        



                i_shift +=1
                if (i_shift==half_FOV*2):
                    i_shift=-half_FOV*2
                left=theta+i_shift*shift_value
                right=theta+(i_shift+1)*shift_value
                if (left<sph[i_pt][6] and sph[i_pt][6]<right):
                    vec=sph[i_pt:i_pt+1]
                else:
                    vec=np.zeros([0,7])
#         print [left, sph[i_pt][6], right, line, i_shift, vec.shape[0]]
                
    return np.float32(data)

        

def sph_prepare1(sph,half_FOV=45):
#     half_FOV=45
    data=np.zeros([64,half_FOV*4,14])

    x_shift_value = 0.5*np.pi/180
    y_shift_value = 0.44*np.pi/180
    for i_pt in range(len(sph)):
        
        # get coordinate
        x1=np.floor(sph[i_pt][6]/x_shift_value)+half_FOV*2
        y1=np.floor((4.2*np.pi/180-sph[i_pt][5])/y_shift_value)
#         print [sph[i_pt][6],x1,sph[i_pt][5],y1]        
        
    
        # if point is in ROI, insert point
        if(x1>=0 and y1>=0 and x1<half_FOV*4 and y1<64):
            x1=np.uint(x1)
            y1=np.uint(y1)
            # if it is minimum or void, insert it
            if ((sph[i_pt,4]<data[y1,x1,4]) or (data[y1,x1,0]==0 and data[y1,x1,1]==0)):
                data[y1,x1,0:7]=sph[i_pt]
            if ((sph[i_pt,4]>data[y1,x1,11]) or (data[y1,x1,0]==0 and data[y1,x1,1]==0)):
                data[y1,x1,7:14]=sph[i_pt]      
    return np.float32(data)

def sph_prepare2(sph,half_FOV=45):
    data=np.zeros([64,half_FOV*4,14])

    x_shift_value = 0.5*np.pi/180
    
    y1=0
    
    for i_pt in range(len(sph)):
        
        # get coordinate
        x1=np.floor(sph[i_pt][6]/x_shift_value)+half_FOV*2

        # if point is in ROI, insert point
        if(x1>=0 and x1<half_FOV*4):
            x1=np.uint(x1)
            y1=np.uint(y1)
            # if it is minimum or void, insert it
            if ((sph[i_pt,4]<data[y1,x1,4]) or (data[y1,x1,0]==0 and data[y1,x1,1]==0)):
                data[y1,x1,0:7]=sph[i_pt]
            if ((sph[i_pt,4]>data[y1,x1,11]) or (data[y1,x1,0]==0 and data[y1,x1,1]==0)):
                data[y1,x1,7:14]=sph[i_pt]    
                
#             print (sph[i_pt+1:i_pt+5,6])    
            if (i_pt+5<len(sph) and (np.min(sph[i_pt+1:i_pt+5,6])>half_FOV*np.pi/180)):
                y1=y1+1
                
                
        if(y1==64):
            break

            
    return np.float32(data)


def sph_prepare3(sph,half_FOV=45):
#     half_FOV=45
    data=np.zeros([16,half_FOV*4,12,7])
    stack=np.uint8(np.zeros([16,half_FOV*4]))
    
    
    
    x_shift_value = 0.5*np.pi/180
    y_shift_value = 0.44*np.pi/180*4
    for i_pt in range(len(sph)):
        
        # get coordinate
        x1=np.floor(sph[i_pt][6]/x_shift_value)+half_FOV*2
        y1=np.floor((4.2*np.pi/180-sph[i_pt][5])/y_shift_value)
        
        
        
        # if point is in ROI, insert point
        if(x1>=0 and y1>=0 and x1<half_FOV*4 and y1<16):
            x1=np.uint(x1)
            y1=np.uint(y1)
#             print (x1,y1,stack[y1,x1])
            #push point to stack
            
            
            if(stack[y1,x1]<11):
                data[y1,x1,stack[y1,x1],0:7]=sph[i_pt]
            stack[y1,x1]+=1
    return np.float32(data)



def img_label_prepare(data_sph,gt_img,Tr_velo_to_img):
    height = data_sph.shape[0]
    width  = data_sph.shape[1]
    
    label  = np.zeros([height,width])
    
    gt_height=gt_img.shape[0]
    gt_width =gt_img.shape[1]
    
    
#     print gt_img.shape
    
    for i in range(height):
        for j in range(width):

            
            if(data_sph[i,j,2]!=0) :
                velo1=np.zeros([2,4])
                velo1[0,0:3]=data_sph[i,j,0:3]
                velo1[1,0:3]=data_sph[i,j,7:10]
                img_ij=Trans_velo_to_img(Tr_velo_to_img,velo1)
                
#                 print(velo1[0],img_ij[0])
                
                x1=np.int(img_ij[0][0])
                y1=np.int(img_ij[0][1])
                x2=np.int(img_ij[1][0])
                y2=np.int(img_ij[1][1])
#                 print [x1,y1,x2,y2]
                
                if(x1>=0 and x1<gt_width 
                   and y1>=0 and y1<gt_height 
                   and x2>=0 and x2<gt_width 
                   and y2>=0 and y2<gt_height
                   and gt_img[y1,x1]>0 and gt_img[y2,x2]>0):
                        label[i,j]=1
#                         print [i,j,x1,y1,data_sph[i,j,0],data_sph[i,j,1]]
                
                
            

    return label


                
            
            
def img_label_prepare2(data_sph,gt_img,Tr_velo_to_img):
    height = data_sph.shape[0]
    width  = data_sph.shape[1]
    depth  = data_sph.shape[2]
    
    label  = np.zeros([width])
    
    gt_height=gt_img.shape[0]
    gt_width =gt_img.shape[1]
    
    
#     print gt_img.shape
    for j in range(width):
        for i in range(height):
        
            for k in range(depth):
            
                if(data_sph[i,j,k,2]!=0) :
                    velo1=np.zeros([2,4])
                    velo1[0,0:3]=data_sph[i,j,k,0:3]
                    img_ij=Trans_velo_to_img(Tr_velo_to_img,velo1)

    #                 print(velo1[0],img_ij[0])

                    x1=np.int(img_ij[0][0])
                    y1=np.int(img_ij[0][1])

                    if(x1>=0 and x1<gt_width 
                       and y1>=0 and y1<gt_height
                       and gt_img[y1,x1]>0
                       and data_sph[i,j,k,4]>label[j]):
                            label[j]=data_sph[i,j,k,4]
#                             print [i,j,k,x1,y1,data_sph[i,j,0],data_sph[i,j,1]]
    return label
            
def BEV_label_prepare(data_sph,gt_BEV_img):
    height = data_sph.shape[0]
    width  = data_sph.shape[1]
    
    label  = np.zeros([height,width])
    
    gt_BEV_array=np.array(gt_BEV_img)
    print gt_BEV_array.shape
    
    for i in range(height):
        for j in range(width):
            x1 = np.int(data_sph[i,j,0]*20)-120
            y1 = np.int(data_sph[i,j,1]*20)+200

            
            x2 = np.int(data_sph[i,j,7]*20)-120
            y2 = np.int(data_sph[i,j,8]*20)+200
            
#             print data_sph[i,j,0]
#             print gt_BEV_array[x1][y1]
            print [data_sph[i,j,1],y1]
            if(x1>=0 and x1<800 
               and y1>=0 and y1<400 
               and x2>=0 and x2<800 
               and y2>=0 and y2<400
               and ~(data_sph[i,j,0]==0 and data_sph[i,j,1]==0)):
                if(gt_BEV_array[x1,y1]>0 and gt_BEV_array[x2,y2]>0):
                    label[i,j]=1
    return label               