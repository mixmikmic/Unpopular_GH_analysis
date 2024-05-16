#First let's import the necessary library.
get_ipython().magic('matplotlib inline')
from Compare import Compare
from json import dumps

# Let's test the class with the new dictionary.  When you have a dictionary with two keys, 
# the Compare class will
# generate a simple venn diagram.

x = Compare({"happy": ["ecstatic", "bursting", "nostalgic"], "sad": ["morose", "depressed", "nostalgic"]},
            LABEL_BOTH_FACTORS=True)

print (x.LABEL_BOTH_FACTORS)

# Here are Committees of the current (2015-2016) sitting parliament.
# 
# There are two sets of variables.  1. MPs and 2. Committees that the MPs belong to.
# Or worded in another way, we are examining how MPs are affiliated with different groups.


Committees = dict({"ESPE": #Pay equity
                   [("Lib", "A Vandenbeld"), ("Con", "S Stubbs"), ("NDP", "S Benson"), ("Con", "D Albas"), 
                    ("Lib", "M DeCourcey"), ("Lib", "J Dzerowicz"), ("Con", "M Gladu"), ("Lib", "T Sheehan"), 
                    ("Lib", "S Sidhu")],
                   
                   "FEWO": #Status of Women
                   [("Con", "M Gladu"), ("Lib", "P Damoff"), ("NDP", "S Malcomson"), ("Lib", "S Fraser"), 
                    ("Con", "R Harder"), ("Lib", "K Ludwig"), ("Lib", "E Nassif"), ("Lib", "R Sahota"), 
                    ("Lib", "A Vandenbeld"), ("Con", "K Vecchio")],
                   
                   "HESA": #Health
                   [("Lib", "B Casey"), ("Con", "L Webber"), ("NDP", "D Davies"), ("Lib", "R Ayoub"), 
                    ("Con", "C Carrie"), ("Lib", "D Eyolfson"), ("Con", "R Harder"), ("Lib", "DS Kang"), 
                    ("Lib", "J Oliver"), ("Lib", "S Sidhu")],
                  
                   "BILI": #Library of Parliament
                   [("Con", "G Brown"), ("Con", "K Diotte"), ("Con", "T Doherty"), ("Lib", "A Iacono"),
                   ("Con", "M Lake"), ("Lib", "M Levitt"), ("Lib", "E Nassif"), ("NDP", "AMT Quach"), 
                   ("Lib", "D Rusnak"), ("Lib", "M Serré"), ("Lib", "G Sikand"), ("Lib", "S Simms")],
                   
                   "RNNR": #Natural Resources
                   [("Lib", "J Maloney"), ("Con", "J Barlow"), ("NDP", "R Cannings"), ("Con", "C Bergen"),
                   ("Lib", "TJ Harvey"), ("Lib", "D Lemieux"), ("Lib", "MV Mcleod"), ("Lib", "M Serré"), 
                   ("Con", "S Stubbs"), ("Lib", "G Tan")],
                   
                   "ACVA": #Veteran's Affairs
                   [("Lib", "NR Ellis"), ("Con", "R Kitchen"), ("NDP", "I Mathyssen"), ("Lib", "B Bratina"),
                   ("Con", "A Clarke"), ("Lib", "D Eyolfson"), ("Lib", "C Fraser"), ("Lib", "A Lockhart"), 
                   ("Lib", "S Romando"), ("Con", "C Wagantall")],
                   
                   "JUST": #Justice and Human Rights
                   [("Lib", "A Housefather"), ("Con", "T Falk"), ("NDP", "M Rankin"), ("Lib", "C Bittle"), 
                    ("Con", "M Cooper"), ("Lib", "C Fraser"), ("Lib", "A Hussen"), ("Lib", "I Khalid"), 
                    ("Lib", "R McKinnon"), ("Con", "R Nicholson")],
                   
                   "TRAN": #Transport, Infrastructure and Communities
                   [("Lib", "JA Sgro"), ("Con", "L Berthold"), ("NDP", "L Duncan"), ("Lib", "V Badawey"), 
                    ("Con", "K Block"), ("Lib", "S Fraser"), ("Lib", "K Hardie"), ("Lib", "A Iacono"), 
                    ("Lib", "G Sikand"), ("Con", "DL Watts")],
                   
                   "AGRI": #Agriculture and Agri-food
                   [("Lib", "P Finnigan"), ("Con", "B Shipley"), ("NDP", "RE Brosseau"), ("Con", "D Anderson"),
                   ("Lib", "P Breton"), ("Lib", "F Drouin"), ("Con", "J Gourde"), ("Lib", "A Lockhart"), 
                    ("Lib", "L Longfield"), ("Lib", "J Peschisolido")],
                   
                   "FOPO": #Fisheries and Oceans
                   [("Lib", "S Simms"), ("Con", "R Sopuck"), ("NDP", "F Donnelly"), ("Con", "M Arnold"), 
                    ("Con", "T Doherty"), ("Con", "P Finnigan"), ("Lib", "K Hardie"), ("Lib", "B Jordan"),
                   ("Lib", "K McDonald"), ("Lib", "RJ Morrissey")]
                  })

# Let's just start with a case of two committees and focus just on the people (not their parties)
EquityStatus = {"ESPE": [y for x, y in Committees["ESPE"]], "FEWO": [y for x, y in Committees["FEWO"]]}
equityStatus = Compare(EquityStatus)

print("There are "+ str(len(equityStatus.V2_AB)) + " members in common. They are : " + str(equityStatus.V2_AB))

# Now let's see what ca does:
LABEL_FOR_TWO = False
Committee = dict()

for com, members in Committees.items():
    Committee[com] = [y for x, y in members]
committees_ca = Compare(Committee, LABEL_BOTH_FACTORS=True)




