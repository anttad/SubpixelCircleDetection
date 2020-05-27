#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import os
import numpy as np 
import cv2
import json
from skimage import  color #, data 
from skimage.transform import hough_circle, hough_circle_peaks,hough_ellipse 
from skimage.feature import canny
from skimage.draw import circle_perimeter, rectangle_perimeter,ellipse_perimeter,circle
from skimage.util import img_as_ubyte

import scipy.misc
import scipy.ndimage
import skimage.morphology
from skimage.morphology import disk
from skimage.viewer import ImageViewer
from utils import *

from skimage.draw import rectangle

from skimage.morphology import binary_closing, binary_dilation, erosion, dilation
from skimage.morphology import square
import skimage.io

import scipy.misc
import scipy.ndimage
from skimage.viewer import ImageViewer
from skimage.color import rgb2gray

import argparse

from utils import * 

from skimage.filters import threshold_otsu

from apply_tophat import minimum_of_directional_tophat_bottomhat




"""
Formule pour calculer l'air d'un polygone connaissant les coordonnées de ses sommets. "shoelace forumula"
"""
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


"""
A partir d'une liste de edge point obtenu par l'algorithme canny-devernay, renvoie
l'image binaire des contours
"""
def get_edge_map(txt,im_dim, width):
    
    coord = np.loadtxt(txt)
    coord = coord[coord[:,0]!=-1,:] # on élimine la délimitation
    A = np.uint(np.round(coord/width)) # on augmente la valeur des coordonnées par 2 et on arrondie pour pouvoir les plascer
    
    new_dim = ( int(im_dim[0]/width), int(im_dim[1]/width))
    edge_map = np.zeros(new_dim)
   
    y,x= tuple(A.T) # corrige l'inversion des coordonnées x et y
    edge_map[(x,y)]=1
    
    return edge_map



"""
determine si un segment de contours est fermé 
"""
def is_closed(segment): 

   x_o, y_o = segment[0,:] 
   x_f, y_f = segment[-1,:] 
   
   if (x_o == x_f) and (y_o == y_f):
       return True
   else: 
       return False
    
"""
renvoie la liste des contours fermés
"""
def get_closed_contour_map(txt,im_dim, width):
    
    coord = np.loadtxt(txt)/width 
    list_of_segment = np.split(coord,np.argwhere(coord[:,0]<0).reshape(-1))
    list_of_segment = [x[1:,:] if x[0,0] <0 else x for x in list_of_segment[:-1]] # le dernier terme de la liste ne sert à rien
    
    
    closed_edge_list = [ np.uint(np.round(segment)) for segment in list_of_segment if is_closed(segment)]
    #coord = coord[coord[:,0]!=-1,:] # on élimine la délimitation
    # on augmente la valeur des coordonnées par 2 et on arrondie pour pouvoir les placer
    # découpage des segments de contours
    new_dim = ( int(im_dim[0]/width), int(im_dim[1]/width))
    
    edge_map = np.zeros(new_dim)
    
    for seg in closed_edge_list:
        y,x= tuple(seg.T) # corrige l'inversion des coordonnées x et y
        edge_map[(x,y)]=1
    return edge_map
        

def svg2png(in_svg, out_png, out_shape):
    import cairo
    import rsvg
    nrow, ncol = out_shape
    img = cairo.ImageSurface(cairo.FORMAT_ARGB32, ncol,nrow)
    
    ctx = cairo.Context(img)
    
    ## handle = rsvg.Handle(<svg filename>)
    # or, for in memory SVG data:
    handle= rsvg.Handle(in_svg)
    
    handle.render_cairo(ctx)
    
    img.write_to_png(out_png)
        
    

def fftzoom(img, factor=2):
    
    if len(img.shape) == 2:
        nrow,ncol = img.shape
        r_alpha, c_alpha= nrow*(factor-1),ncol*(factor-1)
        r_step, c_step = int(np.ceil(r_alpha/2)), int(np.ceil(c_alpha/2))
        fft_im = np.fft.fftshift(np.fft.fft2(img))
        fft_pad = np.pad(fft_im,((r_step,r_step), (c_step, c_step)))
        res = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_pad)))
#        res = np.uint8(255*(res-res.min())/(res.max()-res.min()))
        
    elif len(img.shape)==3:
        nrow,ncol,_ = img.shape
        r_alpha, c_alpha= nrow*(factor-1),ncol*(factor-1)
        r_step, c_step = int(np.ceil(r_alpha/2)), int(np.ceil(c_alpha/2))
        res_list =[]
        for i in range(3):
            fft_im = np.fft.fftshift(np.fft.fft2(img[:,:,i]))
            fft_pad = np.pad(fft_im,((r_step,r_step), (c_step, c_step)))
            res_list.append(np.abs(np.fft.ifft2(np.fft.ifftshift(fft_pad))))
        res= np.stack(res_list, axis=2)
#        res = np.uint8(255*(res-res.min())/(res.max()-res.min()))
        
    return res

    
#%% test 
    
import matplotlib.pyplot as plt

from compare import precision_recall



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', default="./data/50SQE_2018_12_10_0_012.jpeg", type=str,help="path to the input image")
    parser.add_argument('-z',"--zoom", action='store_true', help="(preprocessing) apply 2x fft zoom to the image")
    parser.add_argument('-t',"--top_hat", action='store_true', help="(preprocessing) apply the top-hat procedure to the image")
    parser.add_argument('-k',"--top_hat_size", default=5,type=int , help="(preprocessing) top-hat parmameter")
    parser.add_argument('-o','--output',default="./results/", type=str,help="output folder")
    parser.add_argument('-a',"--auto_th", action='store_true', help="set automatically the threshold using Otsu's histogram method")
    parser.add_argument('-lt','--low',default=5, type=int,help="low threshold for Canny-Devernay's edge extraction")
    parser.add_argument('-ht','--high',default=15, type=int,help="high threshold for Canny-Devernay's edge extraction")
    parser.add_argument('-s','--std',default=0, type=float,help="std Gaussian kernel for Canny-Devernay's edge extraction pre-processing")
    parser.add_argument('-e', '--eval', action="store_true", help="Use only if ground truth is avaible to evaluate the method performance")

    args = parser.parse_args()
    
    gts = np.load('./data/gt.npz')['points']
    path = "./data/"

    im_name = args.input.split('/')[-1].split('.')[0] + '.pgm'
    #im_name= '50SQE_2018_12_10_0_012.pgm'

    im_path = os.path.join(path, im_name)
    test_path = "./results/edges"

    jpeg_file = os.path.join(path,'50SQE_2018_12_10_0_012.jpeg')
    
    img = skimage.io.imread(args.input)
    
#    os.listdir()
    
    
    devernay = "./C/devernay_1.0/devernay"
    
    tophat=args.top_hat
    zoom = args.zoom
    th_size = args.top_hat_size
    
#    zoom=False
    std = args.std
    l_th = args.low
    h_th = args.high
    th_size = args.top_hat_size
    width = 1 + zoom
    
#    std_list =  [0,0.1,0.3,0.5,0.8,1]
#    l_th_list = [0,2,5,7,10,12,15]
#    h_th_list = [5,7,10,12,15,20,30,50]
#    iso_th_list = np.linspace(0.7,0.99,50) # [0.7,0.75,0.8,0.85,0.9,0.91,0.92,0.93,0.94,0.95]
#    zoom_list = [False,True]
#    tophat_list = [True,False]
#    th_size_list = [7,9,11,15,19,21,23]
    

#    th_size_list = [7,9,11,15,19,21,23]
    
    #json_file = os.path.join(json_path,'50SQE_2018_12_10_0_012.json')
    #data = read_json(json_file)
#    TPR = true_positive_rate(circles, data)
    
    
    prec_list = []
    rec_list = []
    f1_list = []
    best_f1 = 0
    im_dim=(img.shape[0],img.shape[1])
    
    
        
    prec_list = []
    rec_list = []
    f1_list = []
    best_f1 = 0
    best_prec =0
    best_rec= 0    
    
#    
#    for zoom in zoom_list:
#        for l_th in l_th_list:
#            for h_th in h_th_list:
#                if l_th< h_th :
#                    for th_size in th_size_list:
#                        print(zoom,l_th,h_th,th_size)
    if zoom : 
        im_path = "./results/tmp/zoom_50SQE_2018_12_10_0_012.pgm"
        img_zoom = fftzoom(img)
        print(img_zoom.shape)
        if tophat :
            img_zoom,img_zoom_gr = minimum_of_directional_tophat_bottomhat(img_zoom,th_size)
            skimage.io.imsave(im_path,img_zoom_gr)
        else : 
            img_zoom_gr = skimage.color.rgb2gray(img_zoom)
            skimage.io.imsave(im_path,img_zoom_gr)
    else : 
        if tophat :
            
            im_path = "./results/tmp/top_hat_50SQE_2018_12_10_0_012.pgm"
            img_zoom,img_zoom_gr = minimum_of_directional_tophat_bottomhat(img,th_size)
            skimage.io.imsave(im_path,img_zoom_gr)
        else : 
            im_path = os.path.join(path, im_name)
            img_zoom = img.copy()
            gr_img = rgb2gray(img_zoom)
            skimage.io.imsave(im_path,gr_img)
            
        
    isprs_img = "./results/tmp/im_tophat_{}_zoom_{}.jpeg".format(tophat,zoom)
    skimage.io.imsave(isprs_img,img_zoom)
    
    
        
    if args.auto_th:
        h_th = int(threshold_otsu(skimage.io.imread(im_path)))
        l_th = 0.5*h_th
    txt_path = os.path.join(test_path,'h_{}_l_{}_sig_{}_zoom_{}_tophat_{}.txt'.format(h_th,l_th,std,int(zoom),int(tophat)))
    pdf_path = os.path.join(test_path,'h_{}_l_{}_sig_{}_zoom_{}_tophat_{}.pdf'.format(h_th,l_th,std,int(zoom), int(tophat)))
    svg_path = os.path.join(test_path,'h_{}_l_{}_sig_{}_zoom_{}_tophat_{}.pdf'.format(h_th,l_th,std,int(zoom), int(tophat)))
    os.system('{} {} -t {} -p {} -s {} -l {} -h {} -w 0.5 '.format(devernay,im_path, txt_path,pdf_path,std,l_th, h_th))
    
    im_dim = (img_zoom.shape[0], img_zoom.shape[1])
    edge_map = get_edge_map(txt_path,im_dim,width)
    
    hough_radii = [i for i in range(5,int(20/width))]

    
    plt.close('all')
    
    
    #cv2
    ################################# Houghs circles ###############################
    hough_radii = [i for i in range(5,int(20/width))]
    
    output = img.copy()
    X,Y,_= img.shape
    newX, newY = X/width, Y/width
    output = cv2.resize(output,(int(newX),int(newY)))
    hough_res = hough_circle(edge_map, hough_radii,full_output = False)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=30)
    
    output = img_zoom.copy()
    centers = np.zeros((img.shape[0],img.shape[1]))
    mask = np.zeros((img.shape[0],img.shape[1]))
    list_center=[]
        
    if zoom :
        for center_y, center_x, radius in zip(cy,cx,radii):
            circy, circx = circle_perimeter(center_y,center_x, radius, shape=output.shape)
            cy, cx = circle(center_y/2,center_x/2, int(np.ceil(radius/2)), shape=output.shape)
            output[circy,circx] = (0,255,0)
            centers[int(center_y/2),int(center_x/2)] = 1
            mask[cy,cx] = 1
            list_center.append((int(center_x/2),int(center_y/2)))
    else : 
        for center_y, center_x, radius in zip(cy,cx,radii):
            circy, circx = circle_perimeter(center_y,center_x, radius, shape=output.shape)
            cy, cx = circle(center_y,center_x, radius, shape=output.shape)
            output[circy,circx] = (0,255,0)
            centers[int(center_y),int(center_x)] = 1
            mask[cy,cx] = 1
            list_center.append((center_x,center_y))
            

        res_img = "./results/HCT/output/detection_mask_zoom_{}_tophat_{}_autoth.png".format(int(zoom), int(tophat))
        skimage.io.imsave(res_img,mask)
        print('image saved !')
       

    
