import os
from dgl.data.utils import load_graphs

for root, _, files in os.walk(
        "J:\\HSG\\ConGNN-SUM\\cache\\CNNDM\\graphs\\test"):
    indexes=[]
    for file in files:
        from_index = int(file[:-4])
        indexes.append(from_index)
        path = os.path.join(root, file)
        # if file=="288512.bin":
        # g, label_dict = load_graphs(path)
        # print(g)
            # print(file,"  ", len(g))
    max_indexes=max(indexes)
    print(max_indexes," ",max_indexes+256)
    for i in range(0,max_indexes,256):
        if i not in indexes:
            print("ERROR: ",i)



