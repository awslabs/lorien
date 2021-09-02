# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The module to interact with DynamoDB.
"""
from typing import Any, Dict, Generator, Optional, List

import boto3

from ..logger import get_logger

log = get_logger("Database")


def create_table(table_name: str, **db_kwargs) -> str:
    """Create an empty table in the DynamoDB if the table does not exist.

    Parameters
    ----------
    table_name: str
        The table name.

    **db_kwargs
        The kwargs of boto3 client. Commonly used: "region_name='us-west-1'"
        or "endpoint_url=http://localhost:8000".

    Returns
    -------
    arn: str
        The table ARN (Amazon Resource Name).
    """

    # Check if the table exists.
    try:
        client = boto3.client("dynamodb", **db_kwargs)
        resp = client.describe_table(TableName=table_name)
        log.info("Table %s exists", table_name)
        return resp["Table"]["TableArn"]
    except Exception as err:  # pylint:disable=broad-except
        pass

    # Key attributes in the table.
    attrs = [
        {"AttributeName": "Target", "AttributeType": "S"},
        {"AttributeName": "PrimaryRangeKey", "AttributeType": "S"},
        {"AttributeName": "TargetIDKeys", "AttributeType": "S"},
    ]

    key_schema = [
        {"AttributeName": "Target", "KeyType": "HASH"},
        {"AttributeName": "PrimaryRangeKey", "KeyType": "RANGE"},
    ]

    global_secondary_indexes = [
        {
            "IndexName": "TargetIDKeysIndex",
            "KeySchema": [
                {"AttributeName": "TargetIDKeys", "KeyType": "HASH"},
                {"AttributeName": "PrimaryRangeKey", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
            "ProvisionedThroughput": {"ReadCapacityUnits": 100, "WriteCapacityUnits": 10},
        }
    ]

    try:
        client = boto3.client("dynamodb", **db_kwargs)
        resp = client.create_table(
            TableName=table_name,
            AttributeDefinitions=attrs,
            KeySchema=key_schema,
            GlobalSecondaryIndexes=global_secondary_indexes,
            ProvisionedThroughput={"ReadCapacityUnits": 100, "WriteCapacityUnits": 10},
        )
        log.info("Table %s created successfully", table_name)
        return resp["TableDescription"]["TableArn"]
    except Exception as err:  # pylint:disable=broad-except
        raise RuntimeError("Error creating table %s: %s" % (table_name, str(err)))


def delete_table(table_name: str, **db_kwargs) -> None:
    """Delete the given table in the database.

    Parameters
    ----------
    table_name: str
        The table name in string.

    **db_kwargs
        The kwargs of boto3 client. Commonly used: "region_name='us-west-1'"
        or "endpoint_url=http://localhost:8000".
    """
    client = boto3.client("dynamodb", **db_kwargs)
    client.delete_table(TableName=table_name)
    log.info("Table %s has been deleted", table_name)


def check_table(table_name: str, table_arn: str, **db_kwargs) -> bool:
    """Check if the DynamoDB table name and ARN match the one this worker can access to.

    Parameters
    ----------
    table_name: str
        Table name.

    table_arn: str
        Table Amazon Resource Name.

    **db_kwargs
        The kwargs of boto3 client. Commonly used: "region_name='us-west-1'"
        or "endpoint_url=http://localhost:8000".

    Returns
    -------
    success: bool
        False if the table and ARN does not exist in the DynamoDB.
    """
    try:
        client = boto3.client("dynamodb", **db_kwargs)
        resp = client.describe_table(TableName=table_name)
        return table_arn == resp["Table"]["TableArn"]
    except Exception:  # pylint:disable=broad-except
        return False


def list_tables(**db_kwargs) -> List[str]:
    """List all table names in the database.

    Parameters
    ----------
    **db_kwargs
        The kwargs of boto3 client. Commonly used: "region_name='us-west-1'"
        or "endpoint_url=http://localhost:8000".

    Returns
    -------
    tables: List[str]
        A list of sorted table names.
    """
    try:
        client = boto3.client("dynamodb", **db_kwargs)
        return sorted(client.list_tables()["TableNames"], reverse=True)
    except Exception as err:  # pylint:disable=broad-except
        raise RuntimeError("Failed to fetch the table list: %s" % str(err))


def scan_table(table_name: str, limit: Optional[int] = None, **db_kwargs) -> Generator:
    """Scan a DynamoDB table for all items. Note that DynamoDB only transfers
    at most 1 MB data per query, so you may need to invoke this generator several times
    to get the entire table.

    Parameters
    ----------
    table_name: str
        The target table name to be scanned.

    **db_kwargs
        The kwargs of boto3 client. For example, use "endpoint_url=http://localhost:8000"
        for local DynamoDB.

    Returns
    -------
    gen: Generator
        A generator that yields a scan query response (at most 1 MB).
    """
    scan_kwargs: Dict[str, Any] = {"TableName": table_name}
    if limit is not None:
        scan_kwargs["Limit"] = limit

    try:
        client = boto3.client("dynamodb", **db_kwargs)
        resp = client.scan(**scan_kwargs)
        yield resp
        while "LastEvaluatedKey" in resp:
            resp = client.scan(ExclusiveStartKey=resp["LastEvaluatedKey"], **scan_kwargs)
            yield resp
    except Exception as err:  # pylint:disable=board-except
        raise RuntimeError("Failed to scan table: %s" % str(err))
