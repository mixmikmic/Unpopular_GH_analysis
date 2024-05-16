from __future__ import division, absolute_import, print_function
import os, sys, time, re, json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

imread = plt.imread
def imread8(im_file):
    ''' Read image as a 8-bit numpy array '''
    im = np.asarray(Image.open(im_file))
    return im

def read_png(res):
    img = Image.open(res)
    return np.array(img)

def read_npy(res):
    return np.load(res)

from unrealcv import client
client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
    sys.exit(-1)

res = client.request('vget /unrealcv/status')
# The image resolution and port is configured in the config file.
print(res)

res = client.request('vget /camera/0/object_mask png')
object_mask = read_png(res)
print(object_mask.shape)
res = client.request('vget /camera/0/normal png')
normal = read_png(res)

# Visualize the captured ground truth
"""
plt.imshow(object_mask)
plt.figure()
plt.imshow(normal)
"""

scene_objects = client.request('vget /objects').split(' ')
print('Number of objects in this scene:', len(scene_objects))

data = {}

for obj_id in scene_objects:
    obj_id_parts = obj_id.split('_')
    class_name = obj_id_parts[0]    
    data[obj_id] = class_name

# Write JSON file
import json
with open('neighborhood_classes.json', 'w') as outfile:
    json.dump(data, outfile)

# TODO: replace this with a better implementation
class Color(object):
    ''' A utility class to parse color value '''
    regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')
    def __init__(self, color_str):
        self.color_str = color_str
        match = self.regexp.match(color_str)
        (self.R, self.G, self.B, self.A) = [int(match.group(i)) for i in range(1,5)]

    def __repr__(self):
        return self.color_str

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

id2mask = {}
for obj_id in scene_objects:
    print(obj_id)
    color = id2color[obj_id]
    mask = match_color(object_mask, [color.R, color.G, color.B], tolerance = 3)
    if mask is not None:
        id2mask[obj_id] = mask

# id2mask.keys() are all the matched objects
for idmask in id2mask.keys():
    print(idmask)

# This may take a while
# TODO: Need to find a faster implementation for this

class_groups = {}

# Go through the matched objects
for idmask in id2mask.keys():
    
    # Get the class from the data object we had earlier for json file
    curr_class = data[idmask]
    
    # If the class is not in the class_groups map, add it
    if curr_class not in class_groups:
        class_groups[curr_class] = [] 
        
    # Add the idmask to it's corresponding class
    class_groups[curr_class].append(idmask)

for class_group in class_groups.keys():
    print(class_group)

# Swap colors method
def swap_color(imgarray, source, dest):
    matched_color = match_color(imgarray, [source.R, source.G, source.B])
    imgarray[:,:,:3][matched_color] = [dest.R, dest.G, dest.B]
    return np.array(imgarray)

# Test swap_color
other_mask = np.array(object_mask)

class_color = Color('(R=31,G=31,B=31,A=255)')

current_color = Color('(R=31,G=191,B=31,A=255)')
object_mask = swap_color(object_mask, current_color, class_color)

current_color = Color('(R=95,G=63,B=95,A=255)')
object_mask = swap_color(object_mask, current_color, class_color)

current_color = Color('(R=95,G=127,B=95,A=255)')
object_mask = swap_color(object_mask, current_color, class_color)

# create copy of object_mask
other_mask = np.array(object_mask)

# Normalize by class
for cls in class_groups.keys():
    
    class_base = class_groups[cls][0]
    class_color = id2color[class_base]
    
    print("class: " + cls + ", color: " + str(class_color))
    
    for class_id in class_groups[cls]:
        
        current_color = id2color[class_id]
        print(current_color)
        object_mask = swap_color(object_mask, current_color, class_color)    

# Show before
before = Image.fromarray(other_mask)
before.show()

# Show after
after = Image.fromarray(object_mask)
after.show()

client.disconnect()

