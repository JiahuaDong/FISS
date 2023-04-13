import numpy as np


filename = os.path.join('./reults/seed_2023-ov','2023-03-03_ade_100-10_OURS.csv')

incre = 10
alls = []
with open(filename, 'r') as f:
    for line_index, line in enumerate(f):
        split = line.split(',')
        all = split[-1]
        all = float(all)
        split = split[1:-1]
        aline = [float(i) for i in split if i not in ('x', 'X')]
        
        if len(aline)==151:
            print(np.mean(aline[0:incre+1]))
            for start in range(11,150,incre):
                print(np.mean(aline[start:start+incre]))

            print(all)

