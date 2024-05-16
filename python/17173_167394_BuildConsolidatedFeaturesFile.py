import csv
import GetPropertiesAPI as GP
import importlib
importlib.reload(GP) # un-comment if there are any changes made to API

def buildFeatureFl(inFL,outFL):    
    reader = csv.reader(open(inFL,"r"))
    head = reader.__next__()

    data = {}
    for row in reader:
        data[row[0]] = row[1:]

    # Extracts all the annotation ID's from IBEIS
    aidList = []
    for gid in data.keys():
        aid = GP.getAnnotID(int(gid))
        data[gid].append(aid)

    # Extracts all feature info based on annotation ID's from IBEIS
    for gid in data.keys():
        if data[gid][3] != None:
            aid = data[gid][3]
            spec_text = GP.getImageFeature(aid,"species_texts")
            data[gid].append(spec_text)
            sex_text = GP.getImageFeature(aid,"sex_texts")
            data[gid].append(sex_text)
            est_age = GP.getImageFeature(aid,"age_months_est")
            data[gid].append(est_age)
            exemplar = GP.getImageFeature(aid,"exemplar_flags")
            data[gid].append(exemplar)
            qual_text = GP.getImageFeature(aid,"quality_texts")
            data[gid].append(qual_text)
        else:
            data[gid].append('NULL')
            data[gid].append('NULL')
            data[gid].append('NULL')
            data[gid].append('NULL')
            data[gid].append('NULL')

    # Write all the extracted info to a CSV file
    head += ['ANNOTATION_ID','SPECIES','SEX','AGE_MONTHS','EXEMPLAR_FLAG','IMAGE_QUALITY']
    writeFL = open(outFL,"w")
    writer = csv.writer(writeFL)
    writer.writerow(head)
    for row in data.keys():
        writer.writerow([row] + data[row])
    writeFL.close()

def __main__():
    buildFeatureFl("../data/consolidatedHITResults.csv","../data/consolidatedHITResultsWithInfo1.csv")
    
if __name__ == __main__:
    __main__()

GP.getAnnotID(5381)

gid_aid_map = {}
for gid in range(1,5384):
    gid_aid_map[gid] = GP.getAnnotID(gid)

import json

with open("../data/flickr_zebra_gid_aid_map.json","w") as fl:
    json.dump(gid_aid_map, fl, indent=4)

list(gid_aid_map.values())


aids = [aid for lst in list(gid_aid_map.values()) for aid in lst if len(lst)]

aid_species_map = {aids[i] : features[i] for i in range(len(aids))}

features = GP.getImageFeature(aids, 'species/text')

with open("../data/flickr_zebra_aid_species_map.json", "w") as fl:
    json.dump(aid_species_map, fl, indent = 4)

import UploadAndDetectIBEIS as UD

UD.check_job_status('jobid-5388')

data_dict = {
        'jobid': 'jobid-5388',
    }
response = UD.get('api/engine/job/status', data_dict)

response



