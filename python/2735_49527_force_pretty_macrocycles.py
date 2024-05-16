from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem
from rdkit.Chem import TemplateAlign

IPythonConsole.use_SVG=False
# IPythonConsole.use_SVG=True  ## comment out this line for GitHub

dhz_smiles = 'C[C@H]1C/C=C/[C@H](CCC/C=C/C2=CC(=CC(=C2C(=O)O1)O)O)O'
dhz = Chem.MolFromSmiles(dhz_smiles)  # from smiles

radicicol_smiles = 'C[C@@H]1C[C@@H]2[C@H](O2)/C=C\C=C\C(=O)Cc3c(c(cc(c3Cl)O)O)C(=O)O1'
radicicol = Chem.MolFromSmiles(radicicol_smiles)

my_molecules = [dhz, radicicol]

Draw.MolsToGridImage(my_molecules)

suppl = Chem.SDMolSupplier('Structure3D_CID_5359013.sdf')  # from SDF via pubchem for 3D conformation
radicicol_2 = [mol for mol in suppl][0]
Chem.SanitizeMol(radicicol_2)
radicicol_2

aligner = Chem.MolFromSmarts('[*r14][cR2]2[cR1][cR1](-[#8])[cR1][cR1](-[#8])[cR2]2[#6r14](=[#8])[#8r14][#6r14]')
Chem.GetSSSR(aligner)
AllChem.Compute2DCoords(aligner)

for mol in my_molecules:
    AllChem.GenerateDepictionMatching2DStructure(mol, 
                                                 aligner,
                                                 acceptFailure = True)

highlight_lists = [mol.GetSubstructMatch(aligner) for mol in my_molecules]
Draw.MolsToGridImage(my_molecules,
                     # highlightAtomLists = highlight_lists  ## uncomment this line to show the substructure match
                    )

aligner_bigger = Chem.MolFromSmarts('[*r14][cR2]2[cR1][cR1](-[#8])[cR1][cR1](-[#8])[cR2]2[#6r14](=[#8])[#8r14]@[#6r14]@[#6r14]')
Chem.GetSSSR(aligner_bigger)
AllChem.Compute2DCoords(aligner_bigger)

for mol in my_molecules:
    AllChem.GenerateDepictionMatching2DStructure(mol, 
                                                 aligner_bigger,
                                                 acceptFailure = True)

highlight_lists = [mol.GetSubstructMatch(aligner_bigger) for mol in my_molecules]
Draw.MolsToGridImage(my_molecules,
                     # highlightAtomLists = highlight_lists  ## uncomment this line to show the substructure match
                    )

suppl = Chem.SDMolSupplier('Structure2D_CID_67524.sdf')  # from SDF via pubchem for 3D conformation
ring_templ = [mol for mol in suppl][0]
Chem.GetSSSR(ring_templ)

ring_templ

# identify atoms in the 14-membered ring

ring_info_list = [mol.GetRingInfo() for mol in my_molecules]
mol_atoms_in_rings = [ring_info.AtomRings() for ring_info in ring_info_list]

size_14_rings = []
for rings in mol_atoms_in_rings:
    for ring in rings:
        if len(ring) == 14:
            size_14_rings.append(ring)
            
print(size_14_rings)

for idx, mol in enumerate(my_molecules):
    TemplateAlign.AlignMolToTemplate2D(mol, 
                                       ring_templ, 
                                       match=size_14_rings[idx],
                                       clearConfs=True)

Draw.MolsToGridImage(my_molecules,
                    highlightAtomLists = size_14_rings
                    )

cd_templ = [mol for mol in Chem.SDMolSupplier('chemdraw_template.sdf')][0]

print(Chem.MolToSmarts(cd_templ))

cd_templ

matches = [mol.GetSubstructMatch(cd_templ) for mol in my_molecules]
print(matches)

for idx, mol in enumerate(my_molecules):
    TemplateAlign.AlignMolToTemplate2D(mol, 
                                       cd_templ, 
                                       match=matches[idx],
                                       clearConfs=True)

Draw.MolsToGridImage(my_molecules,
                    # highlightAtomLists = size_14_or_6_rings
                    )

for atom in radicicol.GetAtoms():
    atom.SetNumExplicitHs(0)
    atom.UpdatePropertyCache()

radicicol

radicicol.GetConformer(0).GetAtomPosition(0)

my_dict = {idx: radicicol.GetConformer(0).GetAtomPosition(idx) for idx in matches[1]}
print(my_dict)

AllChem.Compute2DCoords(radicicol, coordMap = my_dict)



