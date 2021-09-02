"""
This script merges workloads of multiple files.
"""

import argparse

from lorien.util import dump_to_yaml, load_from_yaml
from lorien.workload import Workload


def create_config():
    """Create the config parser."""
    parser = argparse.ArgumentParser(description="Merge Workloads")
    parser.add_argument("-f", "--file", nargs="+", required=True, help="The workload files")
    parser.add_argument(
        "-o", "--output", default="merged_workloads.yaml", help="Output workload file"
    )

    return parser.parse_args()


def main():
    """Main function."""
    configs = create_config()
    workloads = set()

    for file_path in configs.file:
        with open(file_path, "r") as filep:
            row_workloads = load_from_yaml(filep.read())["workload"]
        workloads.update([load_from_yaml(row_workload, Workload) for row_workload in row_workloads])

    out = {"workload": [dump_to_yaml(workload) for workload in workloads]}
    out_str = dump_to_yaml(out, single_line=False)
    with open(configs.output, "w") as filep:
        filep.write(out_str)


if __name__ == "__main__":
    main()
