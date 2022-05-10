import glob
import os
import sys

directory = sys.argv[1]
suboutput = sys.argv[2]

# found = False

for f in glob.glob(directory + "/*/model.pickle", recursive=True):
    # print(f[f.find("\\") + 1:f.rfind("\\")])
    if f[f.find("\\") + 1:f.rfind("\\")] == "alpha_0.05_beta_0.05":
        continue
    # if found:
    for dataset in ["snli", "mnli", "sick"]:
        cmd = "python eval.py --model %s --outputdir %s --dataset %s" % (f, suboutput + f[f.find("\\"):f.rfind("\\") + 1] + dataset, dataset)
        print(cmd)
        os.system(cmd)