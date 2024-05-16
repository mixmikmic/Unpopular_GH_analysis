import csv
import GetPropertiesAPI as GP
import GenerateMTurkFileAPI as GM
import importlib
import random

# un-comment if there are any changes made to API
importlib.reload(GP) 
importlib.reload(GM) 

contributorImages = {}
for contributor in range(1,59):
     contributorImages[contributor] = CB.getContributorGID(contributor)

# Contributors with 0 images
contributorImages.pop(52)
contributorImages.pop(57)
contributorImages.pop(8)
contributorImages.pop(11)
contributorImages.pop(17)
contributorImages.pop(32)
contributorImages.pop(34)
contributorImages.pop(41)

contributors = list(contributorImages.keys())

selectedImgContributors = []
for i in range(100):
    selectedImgContributors.append(contributors[random.randrange(0,50)])

argToAPI = []
for index in selectedImgContributors:
    imgList = contributorImages[index]
    print(len(imgList))
    minGID = min(imgList)
    maxGID = max(imgList)
    argToAPI.append([index,minGID,maxGID])

jobImageMap= {}

for i in range(0,100):
    flName = str("photo_album_%d.question" %(i+1))
    tup = argToAPI[i]
    slctdImgs = GM.generateMTurkFile(tup[1],tup[2],str("/tmp/files/" + flName))
    jobImageMap[flName] = slctdImgs
    i += 1
    print(flName)

len(jobImageMap.keys())



