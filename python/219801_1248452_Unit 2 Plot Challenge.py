import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='chesterish')
get_ipython().run_line_magic('matplotlib', 'inline')

accidents = pd.read_csv("accidents_2012_to_2014.csv")

accidents_severity = accidents.groupby("Accident_Severity")["Accident_Severity"].count()
#When I plotted without the second ["Accident_Severity"] above I seemed to get two lines.
plt.plot(accidents_severity, color='Red')
plt.xticks([1,2,3])
plt.title("Number of Accidents by Severity Type")
plt.ylabel("Count of Accidents")
plt.xlabel("Accident Severity")
plt.show()

plt.figure(figsize=(18, 5))

plt.subplot(1, 2, 1)
plt.hist(accidents.Road_Surface_Conditions)
plt.xticks(rotation=45)
plt.yticks(np.arange(0, 110000, 5000))
plt.title("Road Surface Conditions")
plt.xlabel("Type of Surface Condition")
plt.ylabel("Number of Occurrences")

plt.subplot(1, 2, 2)
plt.hist(accidents.Road_Type)
plt.xticks(rotation=45)
plt.yticks(np.arange(0, 125000, 5000))
plt.title("Road Type")
plt.xlabel("Type of Road")
plt.ylabel("Number of Occurrences")
plt.show()

plt.hist(accidents["Did_Police_Officer_Attend_Scene_of_Accident"])
plt.ylabel("Count")
plt.xlabel("Police Attendance - Yes or No")
plt.title("Police Attendance at Accidents")
plt.show()

plt.scatter(x=accidents["Number_of_Vehicles"], y=accidents["Number_of_Casualties"])
plt.xticks(np.arange(0, 21, 1))
plt.yticks(np.arange(0, 50, 5))
plt.ylabel("Number of Casualties")
plt.xlabel("Number of Cars")
plt.title("Number of Cars vs. Total Casualties")
plt.show()

