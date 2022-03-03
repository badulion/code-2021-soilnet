import argparse
import os
import itertools

COMMAND = """
python run_kube.py \
--job-name soilnet-search-%s-%s \
--priority research-low \
--container-name soilnet-docker \
--image lsx-staff-registry.informatik.uni-wuerzburg.de/dulny/soilnet-feb2022:0.5.0 \
--cpu-limit 16 \
--cpu-request 8 \
--mem-limit 16 \
--mem-request 8 \
--script main.py \
--arguments '-m +experiment=search/%s vars=%s' \
--experiment-path '/home/ls6/dulny/soilnet-Feb2022/' \
--data-mount-dir 'dataset/data/' \
--output-mount-dir 'sweep'

"""

# args list
MODEL = ["rf", "catboost", "mlp", "idw", "knn", "vargp"]
FEATURES = ["metrical", "full", "nobk"]

# config
OUTPUT_FILE = "run_all.sh"


def run_to_str(template, args):
    return template % (args[0], args[1], args[0], args[1])


def main():
    command_list = ["#!/usr/bin/env fish\n"]
    for args in itertools.product(MODEL, FEATURES):
        command_list.append(run_to_str(COMMAND, args))

    full_file = ''.join(command_list)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(full_file)


if __name__ == '__main__':
    main()
