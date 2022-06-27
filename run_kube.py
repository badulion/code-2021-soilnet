import argparse
import os

DEFAULT_CONFIG = {
    'JOB_NAME': 'test-job',
    'PRIORITY': 'research-low',
    'CONTAINER_NAME': 'test',
    'IMAGE': 'test',
    'CPU': '2',
    'MEM': '2Gi',
    'SCRIPT': "test.py",
    'ARGUMENTS': "",
    'DATA_MOUNT_PATH': "~/",
    'OUTPUT_MOUNT_PATH': "~/",
    'LOGNAME': "dulny",
    'USER': "dulny"
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str,
                        help="Name of the job.")
    parser.add_argument("--priority", type=str, default="research-low", choices=["research-low", "research-med", "research-high"],
                        help="Priority of the job.")
    parser.add_argument("--container-name", type=str, default="docker-image",
                        help="Name of the container.")
    parser.add_argument("--image", type=str,
                        help="Path to the image in the registry.")
    parser.add_argument("--cpu", type=int, default=2,
                        help="Number of cpu cores the container will use.")
    parser.add_argument("--mem", type=int, default=4,
                        help="Mmemory the container will use (in GiB).")
    parser.add_argument("--script", type=str,
                        help="The script to run.")
    parser.add_argument("--arguments", type=str, default="",
                        help="Arguments to use with the script (in quotation marks).")
    parser.add_argument("--experiment-path", type=str,
                        help="Root path of the experiment on vingilot.")
    parser.add_argument("--output-mount-dir", type=str,
                        help="Directory within the main experiment directory to mount.")
    parser.add_argument("--data-mount-dir", type=str,
                        help="Directory within the main experiment directory to mount.")
    parser.add_argument("--user-name", type=str,
                        help="Name of the User.")
    parser.add_argument("--log-name", type=str,
                        help="Name of the Log.")
    args = parser.parse_args()
    return args


def main(args):
    with open("template.yml") as f:
        template = f.read()

    template = template.replace("<JOB_NAME>", args.job_name)
    template = template.replace("<PRIORITY>", args.priority)
    template = template.replace("<CONTAINER_NAME>", args.container_name)
    template = template.replace("<IMAGE>", args.image)
    template = template.replace("<CPU>", f"{args.cpu}")
    template = template.replace("<MEM>", f"{args.mem}Gi")
    template = template.replace("<SCRIPT>", args.script)
    template = template.replace("<ARGUMENTS>", str(args.arguments.split()))
    template = template.replace("<EXPERIMENT_PATH>", args.experiment_path)
    template = template.replace("<DATA_MOUNT_PATH>", args.experiment_path+args.data_mount_dir)
    template = template.replace("<OUTPUT_MOUNT_PATH>", args.experiment_path+args.output_mount_dir)

    with open("kubernetes.yml", "w") as f:
        f.write(template)

    os.system(f"kubectl create -f kubernetes.yml")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
