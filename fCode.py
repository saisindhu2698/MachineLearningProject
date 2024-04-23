# In[ ]:

import numpy as np
from IPython import get_ipython
import os as osys
import glob as gl
from random import sample
import sys
import matplotlib.image as mplibimg
import matplotlib.pyplot as pplt
from PIL import Image, ImageChops

# In[ ]:

ipython = get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:

cwd = 'C:\Users\bhano\OneDrive\Desktop\ML\Project\Dataset\archive\'
cwd = osys.getcwd(cwd)
print(osys.listdir('tb_images'))

# In[ ]:

nimg = gl.glob('tb_images/TEST_n*')
pimg = gl.glob('tb_images/TEST_p*')

# In[ ]:

cimgsample = nimg + pimg
print(cimgsample)
nimgsample = sample(nimg, 5)
pimgsample = sample(pimg, 5)

# In[ ]:

cimgfig = pplt.figure(figsize=(15, 8))
cimgfig.suptitle('Lung Tuberculosis dataset', size=18)
for i, imgfname in enumerate(nimgsample + pimgsample):
    sImg = mplibimg.imread(imgfname)
    pplt.subplot(2, 5, i + 1)
    pplt.imshow(sImg, cmap='gray')
    if i <= 4:
        pplt.title('Healthy Patient')
    else:
        pplt.title('Has a Tuberculosis')
pplt.show()

# In[ ]:

osys.chmod('tb_images/afterScaling/',0o777)  
def trim(im):
    bkgImg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diffImg = ImageChops.difference(im, bkgImg)
    diffImg = ImageChops.add(diffImg, diffImg, 2.0, -100)
    bbox = diffImg.getbbox()
    if bbox:
        return im.crop(bbox)

from tqdm import tqdm
print(cimgsample)
for imgfname in tqdm(cimgsample):
    trim(Image.open(imgfname)).save(imgfname.replace('TEST', 'Trimmed'))

# In[ ]:

cimgsample = gl.glob('tb_images/afterScaling/Trim*')

# In[ ]:

trimFig = pplt.figure(figsize=(15, 8))
trimFig.suptitle('Lung Tuberculosis Dataset Trimmed', size=18)
for i, imgfname in enumerate(nimgsample + pimgsample):
    pplt.subplot(2, 5, i + 1)
    pplt.imshow(
        np.array(Image.open(imgfname.replace('TEST', 'Trimmed'))),
        cmap='gray')
    if i <= 4:
        pplt.title('Healthy Patient')
    else:
        pplt.title('Has a Tuberculosis')
pplt.show()

# In[ ]:

osys.chmod('tb_images/afterCompress/',0o777)

for i, imgfname in tqdm(enumerate(cimgsample)):
    Image.open(imgfname).resize((1024, 1024), Image.ANTIALIAS).save(imgfname.replace('Trimmed', 'Compressed'))

# In[ ]:

cimgfig = pplt.figure(figsize=(15, 8))
cimgfig.suptitle('Lung Tuberculosis Dataset Compressed', size=18)
for i, imgfname in enumerate(nimgsample + pimgsample):
    pplt.subplot(2, 5, i + 1)
    pplt.imshow(
        np.array(Image.open(imgfname.replace('CXR_png', 'Compressed'))),
        cmap='gray')
    if i <= 4:
        pplt.title('Healthy Patient')
    else:
        pplt.title('Has a Tuberculosis')

# In[ ]:

import os as osys
import PIL.ImageOps
 
for imgfname in gl.glob('tb_images/afterCompress/*.jpg'):
    PIL.ImageOps.invert(Image.open(imgfname).convert('L')).save('afterInverted/' + osys.path.basename(imgfname))
                                                
    # Plot samples of inverted  images
cimgfig = pplt.figure(figsize=(15, 8))
cimgfig.suptitle('Lung Tuberculosis Dataset Inverted', size=18)
for i, imgfname in enumerate(nimgsample + pimgsample):
    pplt.subplot(2, 5, i + 1)
    pplt.imshow(
        np.array(
            Image.open(imgfname.replace('Compressed', 'Inverted'))), cmap='gray')
    if i <= 4:
        pplt.title('Healthy Patient')
    else:
        pplt.title('Has a Tuberculosis')

# In[ ]:

import glob as gl
import imageio

for image_path in gl.glob("tb_images/TRAIN_n*"):
    nTrain = imageio.imread(image_path)
for image_path in gl.glob("tb_images/TRAIN_p*"):
    pTrain = imageio.imread(image_path)
for image_path in gl.glob("tb_images/TEST_n*"):
    nTest = imageio.imread(image_path)
for image_path in gl.glob("tb_images/TEST_p*"):
    pTest = imageio.imread(image_path)

# In[ ]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# In[ ]:

nTrain = nTrain.reshape(nTrain.shape[0], -1)
pTrain = pTrain.reshape(pTrain.shape[0], -1)
nTest = nTest.reshape(nTest.shape[0], -1)
pTest = pTest.reshape(pTest.shape[0], -1)

# Import the KNeighborsClassifier class from scikit-learn
# In[ ]:

knnAlgo = KNeighborsClassifier()
knnAlgo.fit(nTrain, pTrain)
prdctn = knnAlgo.predict(nTest)

# In[ ]:
trPredict = knnAlgo.predict(nTrain)
print(np.sqrt(mean_squared_error(pTrain, trPredict)))

tePredict = knnAlgo.predict(nTest)
print(np.sqrt(mean_squared_error(pTest, tePredict)))

# Import the DecisionTreeRegressor class from scikit-learn
# In[ ]:

from sklearn.tree import DecisionTreeRegressor

dTreeReg = DecisionTreeRegressor(min_samples_split=8, min_samples_leaf = 80)
fit = dTreeReg.fit(nTrain, pTrain)
prdctn = dTreeReg.predict(nTest)

# In[ ]:

trPredict = dTreeReg.predict(nTrain)
print(np.sqrt(mean_squared_error(pTrain, trPredict)))

tePredict = dTreeReg.predict(nTest)
print(np.sqrt(mean_squared_error(pTest, tePredict)))

# Import the GradientBoostingRegressor class from scikit-learn
# In[ ]:

from sklearn.ensemble import GradientBoostingRegressor

regressor = GradientBoostingRegressor()
regressor.fit(nTrain, pTrain)
prdctn = regressor.predict(X_test)

# In[ ]:

trPredict = regressor.predict(nTrain)
print(np.sqrt(mean_squared_error(pTrain, trPredict)))

tePredict = regressor.predict(nTest)
print(np.sqrt(mean_squared_error(pTest, tePredict)))