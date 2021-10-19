import json

with open('uni_rnd_policy_runs_05.json') as f:
    data = json.load(f)
 
for p in sorted(data.keys()):
    print("p {} r_mean {} r_std {} u_mean {} u_std {}".format(p,data[p]['rnd']['gt_mean'],data[p]['rnd']['gt_std'],data[p]['uni']['gt_mean'],data[p]['uni']['gt_std'] ))
