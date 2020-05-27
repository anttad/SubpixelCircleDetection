import tifffile
import skimage.draw
import imageio
import numpy as np
import scipy.ndimage

def load_image(filepath, bands=['B02', 'B03', 'B04']):
    fim_np = []
    for band in bands:
        im_np = tifffile.imread(filepath.replace('(:band)', band))
        fim_np.append(im_np)
       
    fim_np = np.array(fim_np)
    fim_np = np.moveaxis(fim_np, 0, 2)
    return fim_np
    
def minimum_of_directional_tophat_bottomhat(im_np, size, method='tophat'): # method='tophat', 'bottomhat'
    x = list(range(0, size)) + [size - 1] * size
    y = [0] * size + list(range(0, size))

    fims_np = []
    for i in range(len(x)):
        se_np = np.zeros((size, size), dtype=bool)
        rr, cc = skimage.draw.line(y[i], x[i], size - 1 - y[i], size - 1 - x[i])
        se_np[rr, cc] = True
        
        
        #imageio.imsave('tmp/' + str(i) + '.png', se_np.astype(float))
        
        filtered_np = np.zeros(im_np.shape)
        for j in range(im_np.shape[2]):
            if (method == 'tophat'):
                filtered_np[:,:,j] = im_np[:,:,j] - scipy.ndimage.grey_opening(im_np[:,:,j], size=(size,size), footprint=se_np)
            else:
                filtered_np[:,:,j] = scipy.ndimage.grey_closing(im_np[:,:,j], size=(size,size), footprint=se_np) - im_np[:,:,j]
#        tifffile.imsave('tmp/f_' + str(i) + '.tif', filtered_np)
        fims_np.append(filtered_np)
        
    fims_np = np.array(fims_np)
    fims_np = np.min(fims_np, axis=0)

    fims_grey_np = np.min(fims_np, axis=2)
    #tifffile.imsave('out_grey.tif', fims_np)

    return fims_np, fims_grey_np

#im_np = load_image('data/2019-01-20_S2B_orbit_032_tile_50SNJ_L1C_band_(:band).tif')
#fims_np, fims_grey_np = minimum_of_directional_tophat_bottomhat(im_np, 11)
#tifffile.imsave('tophat_per_channel.tif', fims_np)
#tifffile.imsave('tophat_per_channel_min_all_channels.tif', fims_grey_np)