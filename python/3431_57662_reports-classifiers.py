from classification.dao import ClassifierAccess

from pprint import pprint

from __private import fs

fs.list()

pprint(ClassifierAccess.get_reports(level="alcohol"), indent=4)

pprint(ClassifierAccess.get_reports(level="first_person"), indent=4)

pprint(ClassifierAccess.get_reports(level="first_person_label"), indent=4)









