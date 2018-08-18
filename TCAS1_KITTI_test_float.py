
# coding: utf-8

# In[ ]:


import itertools
import numpy as np
from PIL import Image,ImageDraw
from random import shuffle
from shutil import rmtree
from os import mkdir

import pykitti
import pykitti_obj


from keras.models import load_model, Model


# In[ ]:


# Raw Data directory information
basedir = ''
session = 'KITTI_road/testing/'


res_cam_dir = basedir+session+'res_cam/'
res_BEV_dir = basedir+session+'res_BEV/'
res_vis_dir = basedir+session+'res_vis/'


try:
    rmtree(res_cam_dir)
except OSError:
    pass
try:
    rmtree(res_BEV_dir)
except OSError:
    pass
try:
    rmtree(res_vis_dir)
except OSError:
    pass




mkdir(res_cam_dir)
mkdir(res_BEV_dir)
mkdir(res_vis_dir)



dataset = pykitti_obj.obj(basedir+session)

model_name=basedir+'ChipNet_KITTI2-10layer-float.h5'

model=load_model(model_name)
print model.summary()


# In[ ]:


# assign generator

cam2_iterator   = dataset.cam2
velo_iterator   = dataset.velo
calib_iterator  = dataset.calib
print len(dataset)


# In[ ]:


from skimage import util
from skimage.measure import label ,regionprops
def getLargestCC(segmentation):
    labels = label(segmentation,background=0.0)
    areas=np.bincount(labels.flat)
    areas[0]=0
    
    largestCC = labels == np.argmax(areas)
    return largestCC


# In[ ]:


# for i_sample in range(len(dataset)):
for i_sample in [2]:
    cam2   = next(cam2_iterator)
    velo   = next(velo_iterator)
    calib  = next(calib_iterator)
    sph        = pykitti_obj.xyz2sph(velo)
    data_sph   = pykitti_obj.sph_prepare2(sph,45)
    Tr_velo_to_img = pykitti_obj.read_calib_file(calib)

    
    
    words=calib.replace('/',' ').replace('_',' ').replace('.',' ').split()
#     print words
    res_name=words[-3]+'_road_'+words[-2]+'.png'
#     print res_name
    
    
    
    
    x_dset=np.zeros([1,64,180,14])
    x_dset[0,:,:,:]=data_sph[:,:,:]
       
    
    y_dset=model.predict(x_dset,batch_size=1,verbose=0)

    
    x_shift = 0.5*np.pi/180
    y_shift = 0.44*np.pi/180
    threshold=200
    
    
    y_thr = util.img_as_ubyte(y_dset[0]) > threshold
#     y_clo = binary_closing(y_thr,disk(5))
#     y_clo = binary_opening(y_thr,disk(5))
    y_fil = getLargestCC(y_thr)
    
    boundary=[]
    boundary_xyz=[]




    for j in range(180):
        for i in range(64):
            if (y_fil[i,j]>0 and data_sph[i-1,j,0]>0):
#                 x=(-data_sph[i-1,j,1]+10)*20
#                 y=(data_sph[i-1,j,0]-6)*20
#                 x_velo=data_sph[i-1,j,0]
#                 y_velo=data_sph[i-1,j,1]
#                 z_velo=data_sph[i-1,j,2]
                
                x=(-(data_sph[i-1,j,1]+data_sph[i,j,8])/2+10)*20
                y=((data_sph[i-1,j,0]+data_sph[i,j,7])/2-6)*20
                x_velo=(data_sph[i-1,j,0]+data_sph[i,j,7])/2
                y_velo=(data_sph[i-1,j,1]+data_sph[i,j,8])/2
                z_velo=(data_sph[i-1,j,2]+data_sph[i,j,9])/2
                boundary.append((np.int(x),800-np.int(y)))
                boundary_xyz.append([x_velo,y_velo,z_velo,0])
                break
                
                
                
            elif(y_fil[i,j]>0 and data_sph[i,j,0]>0):
                
#                 x=(-data_sph[i,j,1]*np.tan(data_sph[i,j,5])/np.tan(data_sph[i,j,5]+y_shift)+10)*20
#                 y=(data_sph[i,j,0]*np.tan(data_sph[i,j,5])/np.tan(data_sph[i,j,5]+y_shift)-6)*20
                
#                 x_velo=data_sph[i,j,0]*np.tan(data_sph[i,j,5])/np.tan(data_sph[i,j,5]+y_shift)
#                 y_velo=data_sph[i,j,1]*np.tan(data_sph[i,j,5])/np.tan(data_sph[i,j,5]+y_shift)
#                 z_velo=data_sph[i,j,2]

                x=(-data_sph[i,j,8]*np.tan(data_sph[i,j,12])/np.tan(data_sph[i,j,12]+y_shift/2)+10)*20
                y=(data_sph[i,j,7]*np.tan(data_sph[i,j,12])/np.tan(data_sph[i,j,12]+y_shift/2)-6)*20
                
                x_velo=data_sph[i,j,7]*np.tan(data_sph[i,j,12])/np.tan(data_sph[i,j,12]+y_shift/2)
                y_velo=data_sph[i,j,8]*np.tan(data_sph[i,j,12])/np.tan(data_sph[i,j,12]+y_shift/2)
                z_velo=data_sph[i,j,9]

#                 x=(-data_sph[i,j,8]+10)*20
#                 y=(data_sph[i,j,7]-6)*20
                
#                 x_velo=data_sph[i,j,7]
#                 y_velo=data_sph[i,j,8]
#                 z_velo=data_sph[i,j,9]
                
                
                
                
                boundary.append((np.int(x),800-np.int(y)))
                boundary_xyz.append([x_velo,y_velo,z_velo,0])
#                 print (i,x,y)
#                 print (j,i,data_sph[i,j,1],data_sph[i,j,0],data_sph[i,j,5],data_sph[i,j,5]+y_shift/2,x,y)
                break
    
    
    
    
    lower_boundary=[]
    lower_boundary_xyz=[]
    for j in range(180):
        for i in range(63,-1,-1):
            if(y_fil[i,j]>0 and data_sph[i+2,j,0]>0):
                
                x=(-data_sph[i+1,j,1]+10)*20
                y=(data_sph[i+1,j,0]-6)*20
                
                
                x_velo=data_sph[i+1,j,0]
                y_velo=data_sph[i+1,j,1]
                z_velo=data_sph[i+1,j,2]
                
                
                lower_boundary.append((np.int(x),800-np.int(y)))                                
                lower_boundary_xyz.append([x_velo,y_velo,z_velo,0])
                break


    #### generate res_BEV   #####
#     boundary.append((boundary[-1][0],800))
#     boundary.append((boundary[0][0],800))
    boundary.append((200,920))
    lower_boundary.append((200,920)) 
    
    
    
    res_img=Image.new('L', (400,800), color=0)
    draw = ImageDraw.Draw(res_img)
    draw.polygon(boundary,fill=255,outline=255)
    draw.polygon(lower_boundary,fill=0,outline=0)
    del draw
    res_img.save(res_BEV_dir+res_name,'png')
    
    
    #### generate res_cam   #####
#     boundary_xyz.append([1,boundary_xyz[-1][1],boundary_xyz[-1][2],0])
#     boundary_xyz.append([1,boundary_xyz[0][1],boundary_xyz[0][2],0])

    boundary_xyz.append([1,0,boundary_xyz[0][2],0])
    lower_boundary_xyz.append([1,0,lower_boundary_xyz[0][2],0])

    boundary_img=pykitti_obj.Trans_velo_to_img(Tr_velo_to_img,boundary_xyz) 
    lower_boundary_img=pykitti_obj.Trans_velo_to_img(Tr_velo_to_img,lower_boundary_xyz) 
    
    boundary_img_list=[]
    lower_boundary_img_list=[]
    for j in range(boundary_img.shape[0]):
        boundary_img_list.append((np.int(boundary_img[j,0]),np.int(boundary_img[j,1])))
        lower_boundary_img_list.append((np.int(lower_boundary_img[j,0]),np.int(lower_boundary_img[j,1])))
                                 
#     boundary_img_list.append((np.int(cam2.shape[1]/2),cam2.shape[0]))
#     boundary_img_list.append((np.int(cam2.shape[1]/2),cam2.shape[0]))
                                 
    res_cam=Image.new('L', (cam2.shape[1],cam2.shape[0]), color=0)
    draw_cam = ImageDraw.Draw(res_cam)
    draw_cam.polygon(boundary_img_list,fill=255,outline=255)
    draw_cam.polygon(lower_boundary_img_list,fill=0,outline=0)
    del draw_cam
    res_cam.save(res_cam_dir+res_name,'png')
    
    
    
    
    #### generate res_vis   #####
    res_cam_array=np.array(res_cam)
    vis_cam_array=cam2*255
    vis_cam_array[:,:,0]=vis_cam_array[:,:,0]*(1-res_cam_array/255)
    vis_cam=Image.fromarray(np.uint8(vis_cam_array))
    vis_cam.save(res_vis_dir+res_name,'png')
    
    #### generate res_sph   #####
#     res_sph=Image.fromarray(np.uint8(y_fil)*255)
#     res_sph.save(res_sph_dir+res_name,'png')

                                 
    print "finish  frame "+res_name


# In[ ]:


# boundary


# In[ ]:


# boundary_img


# In[ ]:


vis_cam


# In[ ]:


Image.fromarray(np.uint8(y_dset[0]*255))


# In[ ]:


Image.fromarray(np.uint8(y_fil*255))


# In[ ]:


res_img


# In[ ]:


Image.fromarray(np.uint8(cam2*255))


# In[ ]:


y_dset.shape


# In[ ]:


y_dset[:,:,90]


# In[ ]:


a=util.img_as_ubyte(y_dset[0])
a[:,90]

