"""
The unit test for database.
"""
# pylint:disable=missing-docstring, redefined-outer-name, invalid-name, unused-argument

import boto3
import pytest
from moto import mock_dynamodb2

from lorien.database.table import check_table, create_table, delete_table, list_tables, scan_table
from lorien.database.util import (
    convert_to_db_dict,
    convert_to_db_list,
    convert_to_dict,
    convert_to_list,
)


@mock_dynamodb2
def test_manipulate_table():

    table_name = "lorien-test"

    # Test create table
    with pytest.raises(RuntimeError):
        create_table(table_name, region_name="invalid-region")
    arn = create_table(table_name, region_name="us-west-2")

    # Do not create if the table exists
    arn = create_table(table_name, region_name="us-west-2")

    # Test check table
    assert check_table(table_name, arn, region_name="us-west-2")
    assert not check_table(table_name, arn, region_name="us-west-1")

    # Test list table
    with pytest.raises(RuntimeError):
        list_tables(region_name="invalid-region")
    assert len(list_tables(region_name="us-west-2")) == 1
    assert not list_tables(region_name="us-west-1")

    # Put something in the table
    item = {
        "Target": {"S": "llvm"},
        "TargetIDKeys": {"S": "llvm_cpu"},
        "PrimaryRangeKey": {"S": "key"},
    }
    client = boto3.client("dynamodb", region_name="us-west-2")
    client.put_item(TableName=table_name, Item=item)

    # Test scan table
    scanner = scan_table(table_name, limit=1, region_name="us-west-2")
    count = 0
    while True:
        try:
            next(scanner)
            count += 1
        except StopIteration:
            break
    assert count == 1

    # Remove the unit test table.
    delete_table(table_name, region_name="us-west-2")


def test_database_util():
    orig_list = ["string", 123.34, 345, [1, 2, None], {"a": 2, "b": 8}]
    db_list = convert_to_db_list(orig_list)
    assert orig_list == convert_to_list(db_list)
    with pytest.raises(RuntimeError):
        convert_to_list(db_list["L"])

    orig_dict = {"a": "string", "b": 123.45, "c": None, "d": [1, 2, 3], "e": {"p": 3, "q": 4}}
    db_dict = convert_to_db_dict(orig_dict)
    assert orig_dict == convert_to_dict(db_dict)
    with pytest.raises(RuntimeError):
        convert_to_dict(db_dict["M"])
