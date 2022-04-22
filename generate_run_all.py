import argparse
import os
import itertools

COMMAND = """
python run_kube.py \
--job-name soilnet-evaluation-%s-%s \
--priority research-low \
--container-name soilnet-docker \
--image lsx-staff-registry.informatik.uni-wuerzburg.de/dulny/soilnet-feb2022:latest \
--cpu 8 \
--mem 32 \
--script main.py \
--arguments 'model=best/%s/%s vars=%s +run=range(10) general.save_predictions=true general.study_name=evaluation -m' \
--experiment-path '/home/ls6/dulny/soilnet-Feb2022/' \
--data-mount-dir 'dataset/data/' \
--output-mount-dir 'results'

"""

COMMAND = """
python run_kube.py \
--job-name soilnet-predvis-%s-%s \
--priority research-low \
--container-name soilnet-docker \
--image lsx-staff-registry.informatik.uni-wuerzburg.de/dulny/soilnet-feb2022:latest \
--cpu 8 \
--mem 128 \
--script plot_predictions.py \
--arguments 'model=best/%s/%s vars=%s general.study_name=vis_pred' \
--experiment-path '/home/ls6/dulny/soilnet-Feb2022/' \
--data-mount-dir 'dataset/data/' \
--output-mount-dir 'results'
"""

# args list
MODEL = ["svm", "mlp", "vargp", "rf", "catboost", "idw", "knn"]
MODEL = ["catboost"]
FEATURES = ["metrical", "full", "nobk"]

# config
OUTPUT_FILE = "run_all.sh"


def run_to_str(template, args):
    return template % (args[0], args[1], args[1], args[0], args[1])


def main():
    command_list = ["#!/usr/bin/env fish\n"]
    for args in itertools.product(MODEL, FEATURES):
        command_list.append(run_to_str(COMMAND, args))
    
    #command_list = ["#!/usr/bin/env fish\n"]
    #command_list.append(COMMAND)

    full_file = ''.join(command_list)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(full_file)


if __name__ == '__main__':
    main()
