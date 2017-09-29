import os
import sys
import itertools

dry_run = '--dry-run' in sys.argv
local = '--local' in sys.argv
detach = '--detach' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")

basename = "cmmnist"
grids = [
    {
        'force': [True],
        'no-git': [True],
        'lr': [1e-3],
        'loss': ['bce'],
        'latents': [1],
        'latent-dim': [100],
        'tiny': [True],
        'trans-layers': [2],
        'kl-weight': [1, 0.3, 0.1, 0.03],
    }
]

jobs = []
for grid in grids:
    individual_options = [[{key: value} for value in values]
                          for key, values in grid.items()]
    product_options = list(itertools.product(*individual_options))
    jobs += [{k: v for d in option_set for k, v in d.items()}
             for option_set in product_options]

if dry_run:
    print("NOT starting jobs:")
else:
    print("Starting jobs:")

merged_grid = {}
for grid in grids:
    for key in grid:
        merged_grid[key] = [] if key not in merged_grid
        merged_grid[key] += grid[key]

varying_keys = {key for key in merged_grid if len(merged_grid[key]) > 1}

for job in jobs:
    jobname = basename
    flagstring = ""
    for flag in job:
        if isinstance(job[flag], bool):
            if job[flag]:
                jobname = jobname + "_" + flag
                flagstring = flagstring + " --" + flag
            else:
                print("WARNING: Excluding 'False' flag " + flag)
        elif flag == 'import':
            imported_network_name = job[flag]
            if imported_network_name in base_networks.keys():
                network_location = base_networks[imported_network_name]
                flagstring = flagstring + " --" + \
                    flag + " " + str(network_location)
            else:
                flagstring = flagstring + " --" + flag + " " + \
                    networks_prefix + "/" + str(imported_network_name)
            if flag in varying_keys:
                jobname = jobname + "_" + flag + str(imported_network_name)
        else:
            flagstring = flagstring + " --" + flag + " " + str(job[flag])
            if flag in varying_keys:
                jobname = jobname + "_" + flag + str(job[flag])
    flagstring = flagstring + " --name " + jobname

    jobcommand = "python pytorch.py" + flagstring
    print(jobcommand)

    if local and not dry_run:
        if detach:
            os.system(jobcommand + ' 2> slurm_logs/' + jobname +
                      '.err 1> slurm_logs/' + jobname + '.out &')
        else:
            os.system(jobcommand)

    else:
        with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
            slurmfile.write("#!/bin/bash\n")
            slurmfile.write("#SBATCH --job-name" + "=" + jobname + "\n")
            slurmfile.write("#SBATCH --output=slurm_logs/" +
                            jobname + ".out\n")
            slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
            slurmfile.write(jobcommand)

        if not dry_run:
            os.system((
                "sbatch -N 1 -c 3 --gres=gpu:1 --mem=60000 "
                "--time=2-00:00:00 slurm_scripts/" + jobname + ".slurm &"))
