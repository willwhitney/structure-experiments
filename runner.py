import os
import sys

dry_run = '--dry-run' in sys.argv
local = '--local' in sys.argv
detach = '--detach' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")

basename = "pred-slurm"
grids = [
    {
        'no-git': [True],
        'seq-len': [5],
        'image-width': [32],
        'adversarial-weight': [1, 0.1, 0.001, 0.0001],
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
                jobname = jobname + "_" + flag + \
                    "_" + str(imported_network_name)
                flagstring = flagstring + " --" + \
                    flag + " " + str(network_location)
            else:
                jobname = jobname + "_" + flag + "_" + str(job[flag])
                flagstring = flagstring + " --" + flag + " " + \
                    networks_prefix + "/" + str(job[flag])
        else:
            jobname = jobname + "_" + flag + "_" + str(job[flag])
            flagstring = flagstring + " --" + flag + " " + str(job[flag])
    flagstring = flagstring + " --name " + jobname

    jobcommand = "python adversarial_main.py" + flagstring
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
                "sbatch -N 1 -c 2 --gres=gpu:1 --mem=8000 "
                "--time=2-00:00:00 slurm_scripts/" + jobname + ".slurm &"))
