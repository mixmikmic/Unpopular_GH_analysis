import ee

ee.Initialize()

landsat=ee.Image('LANDSAT/LC8_L1T_TOA/LC81230322014135LGN00').select(['B4', 'B3', 'B2'])
geometry = ee.Geometry.Rectangle([116.2621, 39.8412, 116.4849, 40.01236]);

config = {
    'image':landsat,
    'region':geometry['coordinates'],
    'folder':'data',
    'maxPixels':10**10,
    'fileNamePrefix:':'testLansat',
}

myTask=ee.batch.Export.image.toDrive(**config)

myTask.start()

myTask.status()

tasks = ee.batch.Task.list()
tasks

