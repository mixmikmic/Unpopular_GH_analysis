import pandas as pd
import numpy as np
from cluster_tree import *

from ete3 import Tree, TreeStyle,TextFace, add_face_to_node
ts = TreeStyle()
ts.show_leaf_name = False
ts.scale =  12

ward_tree = pd.read_pickle("uncollapsed_tree.pkl")
c_labels = pd.read_pickle("c_labels.pkl")
doc_term = pd.read_pickle("Tfidf_Matrix.pkl")
term_list = pd.read_pickle("Feature_List.pkl")

tolerance=-0.3
cmeans_terms,ndocs = get_means(c_labels,doc_term[:25000,:])
collapsed_tree = collapse_label_tree(ward_tree,cmeans_terms,ndocs,tolerance)
name_tree = get_name_tree(collapsed_tree,cmeans_terms,ndocs,term_list)

print(str(name_tree)[-3:])

t=Tree(str(name_tree),format=1)
def my_layout(node):
    F = TextFace(node.name, tight_text=True)
    add_face_to_node(F, node, column=0, position="branch-right")
ts.layout_fn = my_layout

t.show(tree_style=ts)



Tree("(A:1,(B:1,(E:1,D:1)Internal_1:0.5)Internal_2:0.5)Root;", format=1).show()

print(ndocs)

