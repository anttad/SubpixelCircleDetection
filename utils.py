#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:51:50 2019

@author: antoine
"""


import numpy as np 
import cv2 
import os 
import json
import matplotlib.pyplot as plt 
import scipy.misc
import scipy.ndimage
import skimage.morphology
from skimage.morphology import disk
from skimage.draw import rectangle, rectangle_perimeter




        
"""
fonction qui cherche à determiner si un segment fermé est un cercle, en regardant sont rapport d'isopérimétrie, 
le seuil doit être inférieur à 1 (égalité pour les cercles parfaits)
"""

#def get_list_of_edge(txt,closed=True):
#    coord = np.loadtxt(txt)
#    list_of_segment = np.split(coord,np.argwhere(coord[:,0]<0).reshape(-1))
#    list_of_segment = [x[1:,:] if x[0,0] <0 else x for x in list_of_segment[:-1]]
#    if closed:
#        closed_edge_list = [segment for segment in list_of_segment if is_closed(segment)]
#        return closed_edge_list
#    else:
#        return list_of_segment



def get_list_of_edge(txt,closed=True):
    coord = np.loadtxt(txt)
    if len(coord) != 0 :
        list_of_segment = np.split(coord,np.argwhere(coord[:,0]<0).reshape(-1))
        list_of_segment = [x[1:,:] if x[0,0] <0 else x for x in list_of_segment[:-1]]
        
        if closed:
            closed_edge_list = [segment for segment in list_of_segment if is_closed(segment)]
        else:
            closed_edge_list = [segment if is_closed(segment) else close_segment(segment) for segment in list_of_segment]
            
        return closed_edge_list
    
    else : 
        return [] 
    

def stretch_im(im_np):
    im_np = im_np.astype(int)
    im_np *= 3
    im_np[im_np > 255] = 255
    return im_np
    

def tophat(im_np):
    tophat_np = []
    if len(im_np.shape) >= 3: 
        for i in range(im_np.shape[2]):
            tophat_np.append(im_np[:,:,i] - skimage.morphology.opening(im_np[:,:,i], selem=disk(10)))
    else : 
        tophat_np.append(im_np - skimage.morphology.opening(im_np, selem=disk(10))) 
    return np.max(tophat_np, axis=0)
    

def bottomhat(im_np):
    bottomhat_np = []
    if len(im_np.shape) >= 3: 
        for i in range(im_np.shape[2]):
            bottomhat_np.append(skimage.morphology.closing(im_np[:,:,i], selem=disk(10)) - im_np[:,:,i])
    else : 
        bottomhat_np.append(skimage.morphology.closing(im_np, selem=disk(10)) - im_np[:,:,i])
    return np.max(bottomhat_np, axis=0)


def get_other_version(im_name,version='th'):
    return im_name[:-5]+'_{}.png'.format(version)



def read_json(json_file):
    with open(json_file) as f :
        data = json.load(f)
    
    return data

"""
Place les pixels de l'image sur l'interval 0-255
"""
def to_255_pxl(img):
    """
    Retourne une image dont les pixels sont entre 0 et 255
    """
    img = img.astype(float)
    return np.round(255*(img - img.min(axis=(0,1)))/(img.max(axis=(0,1))-img.min(axis=(0,1)))).astype(int)



"""
Formule pour calculer l'air d'un polygone connaissant les coordonnées de ses sommets. "shoelace forumula"
"""
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))



"""
Formule pour calculer l'air d'un polygone connaissant les coordonnées de ses sommets. "shoelace forumula"
"""
def PolyPerimeter(coord):
    coord_roll = np.roll(coord,shift=1, axis=0)
    p = np.sum(np.sqrt(np.sum((coord - coord_roll)**2,axis=1)))
    return p



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
Ferme un contour s'il n'est pas fermé
"""
def close_segment(segment): 

   x_o, y_o = segment[0,:] 
   x_f, y_f = segment[-1,:] 
   
   if (x_o == x_f) and (y_o == y_f):
       return True
   else: 
       final_point = np.array([[x_o,y_o]])
       return np.concatenate([segment,final_point])
   

def select_circles(list_of_edge,threshold=0.8,rad_th=0):
    
    circles = []
    for i in range(len(list_of_edge)):
        segment = list_of_edge[i]
        p = PolyPerimeter(segment[:,:])
        x,y = segment[:,1], segment[:,0]  # correction de l'inversion des coordonnées, on ne prend pas le dernier terme
        a = PolyArea(x,y)
        
        q = 4*np.pi*a/(p**2)
#        print('q is equal to ', q)
        m_x, m_y = np.mean(x), np.mean(y)
#        radius = (np.max(x)-np.min(x) + np.max(y) - np.min(y))/4
        radius = np.sqrt(a/np.pi)
#        if q > 1 :
#            print(" Warning !!! q is greater than 1 at {} !!!".format(i))
        if q >= threshold and radius>rad_th: # and q <=1
                circles.append((m_x,m_y,radius))

        
    return circles
        


def get_ground_truth(img,data):
    """
    input: 
        img: np.array, image
        data: dict, json correspondant à l'image
    """
    gt = data['annotations'] # on extrait les données de détections
    output = img.copy()
    mask = np.zeros_like(output)[:,:,0]
    
    for tank_data in gt:
        bbox = tank_data['bbox']
        rx, ry = rectangle(start=(bbox[1], bbox[0]),extent=(bbox[2], bbox[3]))
        rpx, rpy = rectangle_perimeter(start=(bbox[1], bbox[0]),extent=(bbox[2], bbox[3]))
        output[rpx,rpy] = (250,200,20)
        mask[np.int64(rx),np.int64(ry)] = 1
        
    nb_tanks =len(list(gt))
    
    return output, mask, nb_tanks 


def get_bbox_mask(data,N=513):
    """
    input: 
        img: np.array, image
        data: dict, json correspondant à l'image
    """
    
    try : 
        gt = data['annotations'] # on extrait les données de détections
        N = gt[0]['height'] 
    except : 
        gt = []
        
    mask = np.zeros((N,N))
    
    for tank_data in gt:
        bbox = tank_data['bbox']
        rx, ry = rectangle(start=(bbox[1], bbox[0]),extent=(bbox[2], bbox[3]),shape=(N,N))
        mask[np.int64(rx),np.int64(ry)] = 1
        
    nb_tanks =len(list(gt))
    
    return mask, nb_tanks 
        
        
    
def centers2mask(centers,size):
    """
    input : 
        centers: np.array (N,2) avec N le nombre de centres de tanks détectés
        size: tuple, taille de l'image (2D)
    """
    mask=np.zeros(size)
    for i in range(centers.shape[0]):
        y, x = centers[i]
        mask[int(y), int(x)] = True
    
    return mask

def create_folders(res_folder):
    
    try :
        os.mkdir(res_folder)
    except :    
        pass
    
    
    res_edges_path = os.path.join(res_folder,"edges")
    
    try :
        os.mkdir(res_edges_path)
    except :    
        pass
    
    res_out_path = os.path.join(res_folder,"output")
    res_in_path = os.path.join(res_folder,"input")
    res_svg_tanks_path = os.path.join(res_folder,"output_svg_tanks")
    res_svg_path = os.path.join(res_folder,"output_svg")
    res_svg_clusters_path = os.path.join(res_folder,"output_svg_clusters")
    edges_path = "/home/antoine/Documents/THESE_CMLA/Images/Training_sample/edges"
    
    
    try :
        os.mkdir(res_edges_path)
    except :    
        pass
    
    edge_name_list = os.listdir(res_edges_path)
    
    try :
        os.mkdir(res_folder)
    except :    
        pass
    
    try :
        os.mkdir(res_svg_tanks_path)
    except :    
        pass
    
    try :
        os.mkdir(res_svg_path)
    except :    
        pass
    
    try :
        os.mkdir(res_svg_clusters_path)
    except :    
        pass
    
    try :
        os.mkdir(pgm_path)
    except :    
        pass
    try :
        os.mkdir(res_in_path)
    except :    
        pass
    try :
        os.mkdir(res_out_path)
    except :    
        pass
    try :
        os.mkdir(edges_path)
    except :    
        pass
        
def create_edges(name,devernay,res_edges_path,res_in_path,pgm_path="/home/antoine/Documents/THESE_CMLA/Images/Training_sample/pgm_images"):
    
    
    std = 0
    l_th = 5
    h_th = 15
    th_iso = 0.9
    
    from shutil import copy2
    edge_name_list = os.listdir(res_edges_path)
    
    if name+'.txt' not in edge_name_list:
        im_path = os.path.join(pgm_path,name+'.pgm')
        copy2(im_path, res_in_path)        
        edges_path = "/home/antoine/Documents/THESE_CMLA/Images/Training_sample/edges"
        text_file = os.path.join(edges_path,name+'.txt')
        result_png = os.path.join(res_edges_path,name+'.png')
        result_pkl = os.path.join(res_edges_path,name+'.pkl')
        result_pdf = os.path.join(res_edges_path,name+'.pdf')
    #    result_svg = os.path.join(res_edges_path,name+'.svg')
        os.system('{} {} -p {} -t {} -s {} -l {} -h {} -w 0.5 '.format(devernay,im_path,result_pdf,text_file, std, l_th,h_th))
    print("{} Edge computed".format(name))