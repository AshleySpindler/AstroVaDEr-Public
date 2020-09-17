from astropy.io import fits
import wget
import os
import numpy as np
from multiprocessing import Pool
import cv2
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity


def GalaxyGet(IDs, ras, decs, r90s, sample='Train'):
    def galaxyget(i):
        objid = gz['OBJID'][i]
        if os.path.isfile('SCRATCH/'+sample+'/'+sample+'/galaxy_'+str(objid)+'.png')==True:
            print('done it')
            return
        print(i)
        ra, dec, petro_r90 = str(gz['RA'][i]), str(gz['DEC'][i]), str(gz['PETROR90_R'][i]*0.02)
        wget.download(url='http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra='+ra+'&dec='+dec+'&scale='+petro_r90+'&width=256&height=256',
                      out='Scratch/GalaxyZooImages/galaxy_'+str(objid)+'.jpg')
        img = cv2.imread('SCRATCH/'+sample+'/'+sample+'/galaxy_'+str(objid)+'.jpg')
        img_crop = img[32:224,32:224,:]
        img_gray = rgb2gray(img_crop)
        img_scale = resize(img_gray, (128,128,1))
        img_uint = rescale_intensity(img_scale, out_range=(0,255)).astype('uint8')
        cv2.imwrite('SCRATCH/'+sample+'/'+sample+'/galaxy_'+str(objid)+'.png', img_uint)


#Initiate Parallel Processes
if __name__ == '__main__':
    try:
        gz = fits.open('data/gz2sample.fits.gz')[1].data
    except FileNotFoundError:
        wget.download(url='http://zooniverse-data.s3.amazonaws.com/galaxy-zoo-2/gz2sample.fits.gz',
                      out='data/gz2sample.fits.gz')
        gz = fits.open('data/gz2sample.fits.gz')[1].data
        
    train = np.load('data/TrainGals_IDs.npy')
    test = np.load('data/TestGals_IDs.npy')
    valid = np.load('data/ValidGals_IDs.npy')
    ID, ra, dec, petro_r90 = gz['OBJID'], gz['RA'], gz['DEC'], gz['PETROR90_R']*0.02
    
    # DL train galaxies
    vals = np.where(np.isin(gz['OBJID'], train))[0]
    pool = Pool(processes=4)       #Set pool of processors
    pool.map(GalaxyGet(ID, ra, dec, petro_r90, sample='Train'), vals)      #Call function over iterable

    # DL test galaxies
    vals = np.where(np.isin(gz['OBJID'], test))[0]
    pool = Pool(processes=4)       #Set pool of processors
    pool.map(GalaxyGet(ID, ra, dec, petro_r90, sample='Test'), vals)      #Call function over iterable

    # DL valid galaxies
    vals = np.where(np.isin(gz['OBJID'], valid))[0]
    pool = Pool(processes=4)       #Set pool of processors
    pool.map(GalaxyGet(ID, ra, dec, petro_r90, sample='Valid'), vals)      #Call function over iterable
