"""
This script sorts workloads by their shapes and attributes.
"""

import argparse

from lorien.workload import Workload
from lorien.util import dump_to_yaml, load_from_yaml


def create_config():
    """Create the config parser."""
    parser = argparse.ArgumentParser(description="Sort Workloads")
    parser.add_argument("file", help="The workload file")
    parser.add_argument("-i", action="store_true", help="In-place sorting")

    return parser.parse_args()


def main():
    """Main function."""
    configs = create_config()
    with open(configs.file, "r") as filep:
        row_workloads = load_from_yaml(filep.read())["workload"]

    workloads = [load_from_yaml(row_workload, Workload) for row_workload in row_workloads]
    sorted_workloads = sorted(workloads)
    out = {"workload": [dump_to_yaml(workload) for workload in sorted_workloads]}
    out_str = dump_to_yaml(out, single_line=False)

    if configs.i:
        with open(configs.file, "w") as filep:
            filep.write(out_str)
            filep.write("\n")
    else:
        print(out_str)


if __name__ == "__main__":
    main()
