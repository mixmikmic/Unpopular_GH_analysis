from skimage import io, color, feature, transform
import numpy as np
from skimage.filters import rank
import plotly.plotly as py
import cufflinks as cf # this is necessary to link pandas to plotly
import plotly.graph_objs as go
cf.go_offline()
from plotly.offline import plot, iplot
from skimage.morphology import disk
from matplotlib import pyplot as PLT

rgbImg = io.imread("/Users/sreejithmenon/Dropbox/Social_Media_Wildlife_Census/All_Zebra_Count_Images/8598.jpeg")

hsvImg = color.rgb2hsv(rgbImg)

red = np.array([pix[0] for row in rgbImg for pix in row])
green = np.array([pix[1] for row in rgbImg for pix in row])
blue = np.array([pix[2] for row in rgbImg for pix in row])

hue = np.array([pix[0] for row in hsvImg for pix in row])
saturation = np.array([pix[1] for row in hsvImg for pix in row])
value = np.array([pix[2] for row in hsvImg for pix in row])

y = 0.299 * red + 0.587 * green + 0.114 * blue
(np.max(y) - np.min(y))/np.mean(y)

a = np.array(rgbImg)

a

hue

data = [
    go.Histogram(
    x = hue,
    nbinsx = 12
    )
]

plot(data)

a, _ = np.histogram(saturation, bins=5)
np.std(a)

gray = color.rgb2gray(rgbImg)

gray.shape

glcm = feature.greycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
glcm.shape

entropy = rank.entropy(gray, disk(5))

entropy.shape

feature.hog(resized_img, orientations=8)

resized_img = transform.resize(gray, (600,600))

resized_img.shape

PLT.imshow(flipped)
PLT.show()

left = resized_img.transpose()[:300].transpose()
right = resized_img.transpose()[300:].transpose()

left.shape, right.shape

I = np.identity(600)[::-1]

flipped = I.dot(right)

inner = feature.hog(left) - feature.hog(flipped)

np.linalg.norm(inner)

inner

a =dict(a=1, b = 2, c=3) 
a.update(dict(d = 4))

imgFlNms = ["/Users/sreejithmenon/Dropbox/Social_Media_Wildlife_Census/All_Zebra_Count_Images/%i.jpeg" %i for i in range(1,9405)]

imgFlNms



