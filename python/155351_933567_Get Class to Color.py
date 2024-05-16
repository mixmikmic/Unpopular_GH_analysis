from __future__ import division, absolute_import, print_function
import os, sys, time, re, json
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import StringIO
import json

imread = plt.imread
def imread8(im_file):
    ''' Read image as a 8-bit numpy array '''
    im = np.asarray(Image.open(im_file))
    return im

def read_png(res):
    img = Image.open(StringIO.StringIO(res))
    return np.array(img)

def read_npy(res):
    return np.load(res)

# Matches a color from the object_mask and then returns that region color
def match_color(object_mask, target_color, tolerance=3):
    match_region = np.ones(object_mask.shape[0:2], dtype=bool)
    for c in range(3): # r,g,b
        min_val = target_color[c] - tolerance
        max_val = target_color[c] + tolerance
        channel_region = (object_mask[:,:,c] >= min_val) & (object_mask[:,:,c] <= max_val)
        match_region &= channel_region

    if match_region.sum() != 0:
        return match_region
    else:
        return None
    
# Swap colors method
def swap_color(imgarray, source, dest):
    matched_color = match_color(imgarray, [source.R, source.G, source.B])
    imgarray[:,:,:3][matched_color] = [dest.R, dest.G, dest.B]
    return np.array(imgarray)

from unrealcv import client
client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
    sys.exit(-1)

# Make sure the connection works well

res = client.request('vget /unrealcv/status')
# The image resolution and port is configured in the config file.
print(res)

scene_objects = client.request('vget /objects').split(' ')
print('Number of objects in this scene:', len(scene_objects))

if '257' in scene_objects:
    scene_objects.remove('257')

obj_id_to_class = {}
for obj_id in scene_objects:
    obj_id_parts = obj_id.split('_')
    class_name = obj_id_parts[0]    
    obj_id_to_class[obj_id] = class_name

# Write JSON file
with open('neighborhood_object_ids.json', 'w') as outfile:
    json.dump(obj_id_to_class, outfile)

class Color(object):
    ''' A utility class to parse color value '''
    regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')
    def __init__(self, color_str):
        self.color_str = color_str
        match = self.regexp.match(color_str)
        (self.R, self.G, self.B, self.A) = [int(match.group(i)) for i in range(1,5)]

    def __repr__(self):
        return self.color_str

id2color = {}
with open('id2color.json') as data_file:
    data = json.load(data_file)

for obj_id in data.keys():
    color_map = data[obj_id]
    color_str = '(R=' + str(color_map['R']) + ',G=' +                 str(color_map['G']) + ',B=' + str(color_map['B']) +                 ',A=' + str(color_map['A']) + ')'
    color = Color(color_str)
    id2color[obj_id] = color

id2color = {} # Map from object id to the labeling color
for i, obj_id in enumerate(scene_objects):
    color = Color(client.request('vget /object/%s/color' % obj_id))
    id2color[obj_id] = color
    print('%d. %s : %s' % (i, obj_id, str(color)))

# Convert to serializable json dictionary
serializable_map = {}
for color_id in id2color.keys():
    curr_color = id2color[color_id]
    color_map = {}
    color_map['R'] = curr_color.R
    color_map['G'] = curr_color.G
    color_map['B'] = curr_color.B
    color_map['A'] = curr_color.A
    serializable_map[color_id] = color_map

# Write to JSON
with open('id2color.json', 'w') as outfile:
    json.dump(serializable_map, outfile)

# Map classes to lists of objects
with open('neighborhood_object_ids.json') as data_file:
    obj_id_to_class = json.load(data_file)
    
classes = {}

for obj_id in obj_id_to_class.keys():
    
    curr_class = obj_id_to_class[obj_id]
    if curr_class not in classes:
        classes[curr_class] = []
    
    classes[curr_class].append(obj_id)

# Write classes to json
with open('neighborhood_classes.json', 'w') as outfile:
    json.dump(classes, outfile) 

# Get top 20 classes from json file
with open('class2count.json') as data_file:
    class2count = json.load(data_file)

from operator import itemgetter
class2countlist = sorted(class2count.items(), key=itemgetter(1), reverse=True)

top20 = []
for cls in class2countlist[0:21]:
    top20.append(cls[0])
    
top20tocolor = {}
for cls in top20:
    top20tocolor[cls] = id2color[classes[cls][0]]

serializable_map = {}
for class_id in top20tocolor.keys():
    curr_color = id2color[classes[class_id][0]]
    color_map = {}
    color_map['R'] = curr_color.R
    color_map['G'] = curr_color.G
    color_map['B'] = curr_color.B
    color_map['A'] = curr_color.A
    serializable_map[class_id] = color_map
    
with open('top20tocolor.json', 'w') as outfile:
    json.dump(serializable_map, outfile)

# Normalize using built in API
counter = 0
for curr_class in classes.keys():
    
    
    object_list = classes[curr_class]
    curr_color = id2color[object_list[0]]
    
    for obj_id in object_list:
        
        if curr_class in top20:
            client.request('vset /object/' + obj_id + '/color ' +                        str(top20.index(curr_class)) + ' 0 0')
        
            print(str(counter) + '. vset /object/' + obj_id + '/color ' +                        str(top20.index(curr_class)) + ' 0 0')
        else:
            client.request('vset /object/' + obj_id + '/color 0 0 0')
            print(str(counter) + '. vset /object/' + obj_id + '/color 0 0 0')
    
        counter += 1

class2count = {}
for batch in range(1,101):

    # Get random location
    z = 300
    x = random.randint(-5500, 5500)
    y = random.randint(-5500, 5500)

    # Coordinates x, y, z
    client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                    ' ' + str(z)) 

    # Get 10 shots in a series
    angles = []
    a = 0
    while len(angles) < 20:
        angles.append(a)
        a -= 3

    # Increment height sequentially
    heights = []
    height = 300
    while len(heights) < 20:
        heights.append(height)
        height += 50

    for i,angle in enumerate(angles[1:2]):
        
        print("Batch: " + str(batch) + " , Image: " + str(i))
        
        # x, y, z
        client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                        ' ' + str(heights[i])) 

        # Pitch, yaw, roll
        client.request('vset /camera/0/rotation ' + str(angle) + ' 0 0')


        # Get Ground Truth
        res = client.request('vget /camera/0/object_mask png')
        object_mask = read_png(res)
        
        # Get all classes in image
        print("Getting all object_ids in image....")
        id2mask = {}
        for obj_id in scene_objects:
            color = id2color[obj_id]
            mask = match_color(object_mask, [color.R, color.G, color.B], tolerance = 3)
            if mask is not None:
                id2mask[obj_id] = mask
        
        # Update class frequency map
        for idmask in id2mask.keys():
            curr_class = obj_id_to_class[idmask]
            if curr_class not in class2count:
                class2count[curr_class] = 1
            else: 
                class2count[curr_class] += 1
               
        # Write to file
        normalized_img = Image.fromarray(object_mask)
        grayscale_img = normalized_img.convert('L')

        directory = './batch/round' + str(batch) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        grayscale_img.save('./batch/round' + str(batch) + '/seg' +                             str(i) + '.png')
        
        
        res = client.request('vget /camera/0/lit png')
        normal = read_png(res)
        normal = Image.fromarray(normal)
        
        normal.save('./batch/round' + str(batch) + '/pic' +                             str(i) + '.png')
        
        print("Images written. ")

with open('class2count.json', 'w') as outfile:
    json.dump(class2count, outfile)

