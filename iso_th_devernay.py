#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:24:34 2020

@author: antoine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:21:12 2019

@author: antoine

Code pour devernay avec rapport isopérimétrique
"""

import os
import numpy as np 
import cv2
from skimage.filters import threshold_otsu
from skimage import  color #, data 
from skimage.transform import hough_circle, hough_circle_peaks,hough_ellipse 
from skimage.feature import canny
from skimage.draw import circle_perimeter, rectangle_perimeter,rectangle,circle
from skimage.util import img_as_ubyte

from skimage.morphology import binary_closing, binary_dilation, erosion, dilation
from skimage.morphology import square,disk
import skimage.io
from skimage.color import rgb2gray

import scipy.misc
import scipy.ndimage
from skimage.viewer import ImageViewer

import argparse

from utils import * 

from apply_tophat import minimum_of_directional_tophat_bottomhat

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
        
"""
fonction qui cherche à determiner si un segment fermé est un cercle, en regardant sont rapport d'isopérimétrie, 
le seuil doit être inférieur à 1 (égalité pour les cercles parfaits)
"""
def convert2pgm(im_name,folder):
    
    img = scipy.misc.imread(os.path.join(folder,im_name))
    gr_img = skimage.color.rgb2gray(img)
    name = im_name.split('.')[0]
    pgm_name = name  + '.pgm'
    
    scipy.misc.imsave(os.path.join(folder,pgm_name),gr_img)
    
    return True
    

def get_canny_spx_points(f_name,edges_path = None, std=0,l_th=5, h_th=15,keep_closed=True):
    """
    input: 
        f_name: Nom de fichier de l'image à traiter
        std: écart-type du noyau gaussien
        l_th: low threshold pour le detecteur de Canny
        h_th: high threshold pour le detecteur de Canny
        edges_path: lieu où sont stockés les contours au format .txt
        keep_closed: True si on ne veut que les contours fermés
        
    output: 
        E: Liste des contours détectés (faire attention aux)
    """
    
#    "/home/antoine/Documents/THESE_CMLA/Images/Training_sample/edges"

    if edges_path is None:
        edges_path='../edges'
    
    try: 
        os.mkdir(edges_path)
    except : 
        pass
    
    
    im_name = f_name.split('.')[0]
    txt_name = im_name+'.txt'
    txt_path = os.path.join(edges_path,txt_name)
    if not txt_name in os.listdir(edges_path):
        os.system('devernay {} -t {} -s {} -l {} -h {} -w 0.5 '.format(im_path, txt_path,std,l_th, h_th))

    
    E = get_list_of_edge(txt_path,closed=keep_closed)
    
    return E



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
            res_temp = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_pad)))
            per_inf = np.percentile(res_temp,q=1)
            per_sup = np.percentile(res_temp,q=99)
            res_temp= np.clip(res_temp,per_inf,per_sup)
            res_temp = np.uint8(255*(res_temp-res_temp.min())/(res_temp.max()-res_temp.min()))
            res_list.append(res_temp)
        res= np.stack(res_list, axis=2)
#        res = np.uint8(255*(res-res.min())/(res.max()-res.min()))
        
    return res


def img_dyn_enhancement(img,q_inf=1,q_sup=99):
    
    if len(img.shape)==3:
        for i in range(img.shape[-1]):
            per_inf = np.percentile(img[:,:,i],q=q_inf)
            per_sup = np.percentile(img[:,:,i],q=q_sup)
            new_chan =np.clip(img[:,:,i],per_inf,per_sup)
            img[:,:,i] = (new_chan-new_chan.min())/(new_chan.max()-new_chan.min())         
            
    return img
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
    parser.add_argument('-it', '--iso_th',type=float, default=0.9, help="Isoperimetric threshold")

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
    iso_th = args.iso_th
    width = 1 + zoom
    
#    
#    gts = np.load('/home/antoine/Documents/THESE_CMLA/ISPRS2020/gt.npz')['points']
#    json_path = "/home/antoine/Documents/THESE_CMLA/Images/Training_sample/training"
#    path = "/home/antoine/Documents/THESE_CMLA/Images/Training_sample/pgm_images"
#    
#    
#
#    im_name='50SQE_2018_12_10_0_012.pgm'
#
#    im_path = os.path.join(path, im_name)
#    test_path = '/home/antoine/Documents/THESE_CMLA/Images/test/'
#    edges_path = "/home/antoine/Documents/THESE_CMLA/Images/Training_sample/edges"
#    
#    isprs_fold = "/home/antoine/Documents/THESE_CMLA/ISPRS2020/"
#    jpeg_file = os.path.join(isprs_fold,'50SQE_2018_12_10_0_012.jpeg')
#    
#    img = skimage.io.imread(jpeg_file)
#    
##    os.listdir()
#    
#    
#    devernay = "/home/antoine/Documents/THESE_CMLA/CodeV0/Devernay_Ipol/devernay_1.0/devernay"
#    
#    tophat=True
#    zoom=True
#    std = 0
#    auto_th= True
#    l_th = 2
#    h_th = 7
#    iso_th =0.9
#    th_size = 11
#    
#    std_list =  [0,0.1,0.3,0.5,0.8,1]
#    l_th_list = [2,5,7,10,12,15]
#    h_th_list =  [5,7,10,12,15,20,30,50]
#    iso_th_list = np.linspace(0.7,0.99,10) # 38 
#    zoom_list = [True,False]
#    tophat_list = [True,False]
#    th_size_list = [5,11,15,19,23]
#    
#    json_file = os.path.join(json_path,'50SQE_2018_12_10_0_012.json')
#    data = read_json(json_file)
##    TPR = true_positive_rate(circles, data)
#    
    
    prec_list = []
    rec_list = []
    f1_list = []
    best_f1 = 0
    best_prec = 0
    best_rec = 0    
#    
#    for tophat in tophat_list :

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
        
        
    output = img_zoom.copy()
    
    im_dim=img.shape
    
    #    
    list_of_edge = get_list_of_edge(txt_path,closed=True)
#                        list_of_edge = get_canny_spx_points('h_{}_l_{}_sig_{}_zoom_{}_tophat_{}.txt'.format(h_th,l_th,std,int(zoom),int(tophat)),test_path)
    circles = select_circles(list_of_edge,threshold=iso_th)
    
     
    #    centers=np.array(circles)[:,:-1]
    #    gt_img, gt_mask, nb_tanks_gt = get_ground_truth(img,data)
    #    mask_centers = centers2mask(centers,gt_mask.shape)
    #    TPR = true_positive_rate(mask_centers, gt_mask, nb_tanks_gt)
    
         
    
    centers = np.zeros((img.shape[0],img.shape[1]))
    mask = np.zeros((img.shape[0],img.shape[1]))
    list_center=[]
    if zoom : 
        for center_y, center_x, radius in circles:
            circy, circx = circle_perimeter(int(center_y),int(center_x), int(np.ceil(radius)), shape=output.shape)
            cy, cx = circle(int(center_y),int(center_x), int(np.ceil(radius)), shape=output.shape)
            output[circy,circx] = (0,255,0)
            centers[int(center_y/2),int(center_x/2)] = 1
            mask[np.uint16(np.ceil(cy/2)),np.uint16(np.ceil(cx/2))] = 1
            list_center.append((int(center_x/2),int(center_y/2)))
    
    else:

        for center_y, center_x, radius in circles:
            circy, circx = circle_perimeter(int(center_y),int(center_x), int(np.ceil(radius)), shape=output.shape)
            cy, cx = circle(int(center_y),int(center_x), int(np.ceil(radius)), shape=output.shape)
            output[circy,circx] = (0,255,0)
            centers[int(center_y),int(center_x)] = 1
            mask[cy,cx] = 1
            list_center.append((center_x,center_y))
        
        
#    if len(list_center) == 0:
#        prec=0
#        rec=0
#        f1_score=0
#    else:
#        prec, rec = precision_recall(np.array(list_center),gts)
#        f1_score = 2*prec*rec/(prec+rec)


#                                    cv2.imwrite("/home/antoine/Documents/THESE_CMLA/ISPRS2020/iso_devernay/h_{}_l_{}_sig_{}_th_{}.png".format(h_th,l_th,std,iso_th),output)
    res_img = "./results/iso_th/output/detection_mask_zoom_{}_tophat_{}_autoth.png".format(int(zoom), int(tophat))
    skimage.io.imsave(res_img,mask)
    print('image saved !')
                    




