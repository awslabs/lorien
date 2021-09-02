"""
A script to download the entire DynamoDB table
"""
# pylint: disable=invalid-name

import argparse
import boto3
from lorien import database
import sys, time


def dump_items(fp, items):
    num_written = 0
    for item in items:
        config_list = database.query.convert_to_list(item["BestConfigs"])
        best_result = max(config_list, key=lambda r: r["thrpt"])
        if best_result["thrpt"] == -1:
            continue
        fp.write("{}\n".format(best_result["config"]))
        num_written += 1
    return num_written


def list_table(args):
    print(database.list_tables(""))


def download_table(args):
    db = boto3.client("dynamodb")
    table_name = args.table_name
    limit_of_items = args.limit_of_items
    try:
        table_info = db.describe_table(TableName=table_name)
    except Exception as err:
        raise RuntimeError("Fail to get information of table {}: {}".format(table_name, err))

    if limit_of_items != 0 and table_info["Table"]["ItemCount"] > limit_of_items:
        prompt_txt = (
            "The number of items ({}) in table {} "
            "exceeds the number of items limit {}\n"
            "Download it anyway? (Y/N) ".format(
                table_info["Table"]["ItemCount"], table_name, limit_of_items
            )
        )
        txt = input(prompt_txt)
        if txt.strip().lower() != "y" and txt.strip().lower() != "yes":
            return

    output_file_name = "{}.log".format(table_name)
    with open(output_file_name, "w") as fp:
        num_iter = 0
        num_written = 0
        start_key = None
        query_options = {
            "TableName": table_name,
            "ProjectionExpression": "BestConfigs",
            "Limit": 100,
        }

        while num_iter == 0 or start_key is not None:
            try:
                if start_key is None:
                    ret = db.scan(**query_options)
                else:
                    ret = db.scan(**query_options, ExclusiveStartKey=start_key)
            except Exception as err:
                raise RuntimeError("Fail to download table {}: {}".format(table_name, err))

            items = ret["Items"]
            num_written += dump_items(fp, items)
            start_key = ret["LastEvaluatedKey"] if "LastEvaluatedKey" in ret else None
            num_iter += 1

        print(
            "Finished downloading table {}, {} records written to {}".format(
                table_name, num_written, output_file_name
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download the best configs from a Lorien-format DynamoDB table"
    )
    subparser = parser.add_subparsers(dest="options")
    subparser.required = True

    list_parser = subparser.add_parser("list_table", help="List all DynamoDB tables")
    list_parser.set_defaults(entry_func=list_table)

    download_parser = subparser.add_parser(
        "download_table",
        help="Download the best configs from a Lorien-format DynamoDB table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    download_parser.add_argument(
        "--table_name", "-t", help="The name of the Lorien-format DynamoDB table", required=True
    )
    download_parser.add_argument(
        "--limit_of_items",
        "-l",
        type=int,
        help="The number of items limit of the DynamoDB table that can be downloaded (0 = unlimited)",
        default=10000,
    )
    download_parser.set_defaults(entry_func=download_table)

    args = parser.parse_args()
    args.entry_func(args)


if __name__ == "__main__":
    main()
