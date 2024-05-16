import piexif
#jpg_name = "/home/csherwood/crs/proj/2015-12-09_survey/overwash_channel/DSC_4339.JPG"
jpg_name = "/home/csherwood/crs/proj/2016_CACO/test_images/IMG_0087.JPG"
exif_dict = piexif.load(jpg_name)
for ifd in ("0th", "Exif", "GPS", "1st"):
    print ifd
    for tag in exif_dict[ifd]:
        print(piexif.TAGS[ifd][tag]["name"], exif_dict[ifd][tag])

# munt the GPS data, just for fun...
print exif_dict['GPS']
for tag in exif_dict['GPS']:
        print(piexif.TAGS['GPS'][tag]["name"], exif_dict['GPS'][tag])

