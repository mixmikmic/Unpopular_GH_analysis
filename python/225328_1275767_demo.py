from __future__ import print_function
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    return color_array

# a nice example
key = '_7zhntDU5r1EmkFSuzKxaQ'

# read in config file
with open('config.json') as config_file:
    config = json.load(config_file)
# in this example we are only interested in the labels
labels = config['labels']

print("mapping: ", config["mapping"])
print("version: ", config["version"])
print("folder_structure:", config["folder_structure"])
print("There are {} labels in the config file".format(len(labels)))

for label_id, label in enumerate(labels):
    print("{:>30} ({:2d}): {:<50} has instances: {}".format(label["readable"], label_id, label["name"], label["instances"]))

for label_id, label in enumerate(labels):
    if(label["instances"]):
             print(label["readable"])

# set up paths for every image
image_path = "training/images/{}.jpg".format(key)
label_path = "training/labels/{}.png".format(key)
instance_path = "training/instances/{}.png".format(key)

# load images
base_image = Image.open(image_path)
label_image = Image.open(label_path)
instance_image = Image.open(instance_path)

# convert labeled data to numpy arrays for better handling
label_array = np.array(label_image)

# for visualization, we apply the colors stored in the config
colored_label_array = apply_color_map(label_array, labels)

r = 1050
c = 3600
print('Pixel at [{}, {}] is: {}'.format(r, c, labels[label_array[r, c]]['readable']))

# convert labeled data to numpy arrays for better handling
instance_array = np.array(instance_image, dtype=np.uint16)

# now we split the instance_array into labels and instance ids
instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
colored_instance_label_array = apply_color_map(instance_label_array, labels)

# instance ids
instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)

a = np.array([[1, 2, 3], [3, 2, 1], [4, 3, 2]])
m = np.zeros((3, 3))
m[a == 1] = 1
print(m)

instances = np.unique(instance_array)
class_ids = instances // 256

instaces_count = instances.shape[0]
classes_id_count = np.unique(class_ids).shape[0]

mask = np.zeros([instance_array.shape[0], instance_array.shape[1], instaces_count], dtype=np.uint8)
print("There are {} masks, {} classes in this image".format(instaces_count, classes_id_count))

for i in range(instaces_count):
    m = np.zeros((instance_array.shape[0], instance_array.shape[1]))
    m[instance_array == instances[i]] = 1
    mask[:, :, i] = m
    print('New mask {} created: instance {} of class {}'.format(i, instances[i], labels[class_ids[i]]["readable"]))

r = 2700
c = 1000
print('Pixel at [{}, {}] is labelled: {}, instance: {}'.format(r, c, labels[instance_label_array[r, c]]['readable'], instance_ids_array[r, c]))

print(class_ids)
# plot a mask
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,15))

n = 52
ins = instances[n]
print('Mask {}: instance{} of class {}'.format(n, ins, labels[ins//256]["readable"]))

ax.imshow(mask[:, :, n])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title(labels[ins//256]["readable"])



# plot the result
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,15))

ax[0][0].imshow(base_image)
ax[0][0].get_xaxis().set_visible(False)
ax[0][0].get_yaxis().set_visible(False)
ax[0][0].set_title("Base image")

ax[0][1].imshow(colored_label_array)
ax[0][1].get_xaxis().set_visible(False)
ax[0][1].get_yaxis().set_visible(False)
ax[0][1].set_title("Labels")

ax[1][0].imshow(instance_ids_array)
ax[1][0].get_xaxis().set_visible(False)
ax[1][0].get_yaxis().set_visible(False)
ax[1][0].set_title("Instance IDs")

ax[1][1].imshow(colored_instance_label_array)
ax[1][1].get_xaxis().set_visible(False)
ax[1][1].get_yaxis().set_visible(False)
ax[1][1].set_title("Labels from instance file (identical to labels above)")

fig.savefig('MVD_plot.png')



