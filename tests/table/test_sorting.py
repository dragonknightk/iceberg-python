# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,eval-used
import json
from typing import Any

import pytest

from pyiceberg.exceptions import ValidationError
from pyiceberg.schema import Schema
from pyiceberg.table.metadata import TableMetadata, TableMetadataUtil
from pyiceberg.table.sorting import (
    UNSORTED_SORT_ORDER,
    NullOrder,
    SortDirection,
    SortField,
    SortOrder,
)
from pyiceberg.table.update import AddSortOrderUpdate, SetDefaultSortOrderUpdate, update_table_metadata
from pyiceberg.transforms import BucketTransform, IdentityTransform, VoidTransform, YearTransform
from pyiceberg.types import IntegerType, NestedField, StructType


@pytest.fixture
def sort_order() -> SortOrder:
    return SortOrder(
        SortField(source_id=19, transform=IdentityTransform(), null_order=NullOrder.NULLS_FIRST),
        SortField(source_id=25, transform=BucketTransform(4), direction=SortDirection.DESC),
        SortField(source_id=22, transform=VoidTransform(), direction=SortDirection.ASC),
        order_id=22,
    )


def test_serialize_sort_order_default() -> None:
    assert (
        SortOrder(SortField(source_id=19)).model_dump_json()
        == '{"order-id":1,"fields":[{"source-id":19,"transform":"identity","direction":"asc","null-order":"nulls-first"}]}'
    )


def test_serialize_sort_order_unsorted() -> None:
    assert UNSORTED_SORT_ORDER.model_dump_json() == '{"order-id":0,"fields":[]}'


def test_serialize_sort_order(sort_order: SortOrder) -> None:
    expected = (
        '{"order-id":22,"fields":['
        '{"source-id":19,"transform":"identity","direction":"asc","null-order":"nulls-first"},'
        '{"source-id":25,"transform":"bucket[4]","direction":"desc","null-order":"nulls-last"},'
        '{"source-id":22,"transform":"void","direction":"asc","null-order":"nulls-first"}]}'
    )
    assert sort_order.model_dump_json() == expected


def test_deserialize_sort_order(sort_order: SortOrder) -> None:
    payload = (
        '{"order-id": 22, "fields": ['
        '{"source-id": 19, "transform": "identity", "direction": "asc", "null-order": "nulls-first"}, '
        '{"source-id": 25, "transform": "bucket[4]", "direction": "desc", "null-order": "nulls-last"}, '
        '{"source-id": 22, "transform": "void", "direction": "asc", "null-order": "nulls-first"}]}'
    )

    assert SortOrder.model_validate_json(payload) == sort_order


def test_sorting_schema(example_table_metadata_v2: dict[str, Any]) -> None:
    table_metadata = TableMetadataUtil.parse_raw(json.dumps(example_table_metadata_v2))

    assert table_metadata.sort_orders == [
        SortOrder(
            SortField(2, IdentityTransform(), SortDirection.ASC, null_order=NullOrder.NULLS_FIRST),
            SortField(
                3,
                BucketTransform(4),
                direction=SortDirection.DESC,
                null_order=NullOrder.NULLS_LAST,
            ),
            order_id=3,
        )
    ]


def test_sorting_to_string(sort_order: SortOrder) -> None:
    expected = """[
  19 ASC NULLS FIRST
  bucket[4](25) DESC NULLS LAST
  void(22) ASC NULLS FIRST
]"""
    assert str(sort_order) == expected


def test_sorting_to_repr(sort_order: SortOrder) -> None:
    expected = (
        "SortOrder("
        "SortField(source_id=19, transform=IdentityTransform(), "
        "direction=SortDirection.ASC, null_order=NullOrder.NULLS_FIRST), "
        "SortField(source_id=25, transform=BucketTransform(num_buckets=4), "
        "direction=SortDirection.DESC, null_order=NullOrder.NULLS_LAST), "
        "SortField(source_id=22, transform=VoidTransform(), "
        "direction=SortDirection.ASC, null_order=NullOrder.NULLS_FIRST), "
        "order_id=22)"
    )
    assert repr(sort_order) == expected


def test_unsorting_to_repr() -> None:
    expected = """SortOrder(order_id=0)"""
    assert repr(UNSORTED_SORT_ORDER) == expected


def test_sorting_repr(sort_order: SortOrder) -> None:
    """To make sure that the repr converts back to the original object"""
    assert sort_order == eval(repr(sort_order))


def test_serialize_sort_field_v2() -> None:
    expected = SortField(source_id=19, transform=IdentityTransform(), null_order=NullOrder.NULLS_FIRST)
    payload = '{"source-id":19,"transform":"identity","direction":"asc","null-order":"nulls-first"}'
    assert SortField.model_validate_json(payload) == expected


def test_serialize_sort_field_v3() -> None:
    expected = SortField(source_id=19, transform=IdentityTransform(), null_order=NullOrder.NULLS_FIRST)
    payload = '{"source-ids":[19],"transform":"identity","direction":"asc","null-order":"nulls-first"}'
    assert SortField.model_validate_json(payload) == expected


def test_deserialize_sort_field_empty_source_ids_rejected() -> None:
    payload = '{"source-ids":[],"transform":"identity","direction":"asc","null-order":"nulls-first"}'
    with pytest.raises(Exception, match="Empty source-ids is not allowed"):
        SortField.model_validate_json(payload)


def test_deserialize_sort_field_source_id_and_source_ids_rejected() -> None:
    payload = '{"source-id":5,"source-ids":[19,20],"transform":"bucket[4]","direction":"asc","null-order":"nulls-first"}'
    with pytest.raises(Exception, match="source-id and source-ids are mutually exclusive"):
        SortField.model_validate_json(payload)


def test_incompatible_source_column_not_found(sort_order: SortOrder) -> None:
    schema = Schema(NestedField(1, "foo", IntegerType()), NestedField(2, "bar", IntegerType()))

    with pytest.raises(ValidationError) as exc:
        sort_order.check_compatible(schema)

    assert "Cannot find source column for sort field: 19 ASC NULLS FIRST" in str(exc.value)


def test_incompatible_non_primitive_type() -> None:
    schema = Schema(NestedField(1, "foo", StructType()), NestedField(2, "bar", IntegerType()))

    sort_order = SortOrder(SortField(source_id=1, transform=IdentityTransform(), null_order=NullOrder.NULLS_FIRST))

    with pytest.raises(ValidationError) as exc:
        sort_order.check_compatible(schema)

    assert "Cannot sort by non-primitive source field: 1: foo: optional struct<>" in str(exc.value)


def test_incompatible_transform_source_type() -> None:
    schema = Schema(NestedField(1, "foo", IntegerType()), NestedField(2, "bar", IntegerType()))

    sort_order = SortOrder(
        SortField(source_id=1, transform=YearTransform(), null_order=NullOrder.NULLS_FIRST),
    )

    with pytest.raises(ValidationError) as exc:
        sort_order.check_compatible(schema)

    assert "Invalid source field foo with type int for transform: year" in str(exc.value)


def _commit_stale_sort_order(
    example_table_metadata_v2: dict[str, Any], *added: SortOrder, stale: SortOrder | tuple[SortOrder, ...]
) -> TableMetadata:
    """Another writer adds `added` and resets the default to 3, then a writer that read the fixture commits `stale`."""
    metadata = TableMetadataUtil.parse_raw(json.dumps(example_table_metadata_v2))
    for order in added:
        metadata = update_table_metadata(
            metadata, (AddSortOrderUpdate(sort_order=order), SetDefaultSortOrderUpdate(sort_order_id=-1))
        )
    metadata = update_table_metadata(metadata, (SetDefaultSortOrderUpdate(sort_order_id=3),))
    # These are the updates UpdateSortOrder builds, and AssertDefaultSortOrderId(3) still holds. Several orders stand
    # for several update_sort_order() calls in one transaction, each of which sends its own Add and Set(-1)
    stale_orders = (stale,) if isinstance(stale, SortOrder) else stale
    return update_table_metadata(
        metadata,
        tuple(
            update
            for order in stale_orders
            for update in (AddSortOrderUpdate(sort_order=order), SetDefaultSortOrderUpdate(sort_order_id=-1))
        ),
    )


# The fixture only has sort order 3, so a writer that read it assigns id 4 to a new order
SORT_BY_X = SortOrder(SortField(source_id=1, transform=IdentityTransform()), order_id=4)
SORT_BY_Y = SortOrder(SortField(source_id=2, transform=IdentityTransform()), order_id=4)
SORT_BY_X_DESC = SortOrder(SortField(source_id=1, transform=IdentityTransform(), direction=SortDirection.DESC), order_id=4)


def test_add_sort_order_reuses_equal_order(example_table_metadata_v2: dict[str, Any]) -> None:
    metadata = _commit_stale_sort_order(example_table_metadata_v2, SORT_BY_X, stale=SORT_BY_X)

    # Java's TableMetadata.Builder.addSortOrder reuses order 4 instead of listing it twice
    assert [sort.order_id for sort in metadata.sort_orders] == [3, 4]
    assert metadata.default_sort_order_id == 4


def test_add_sort_order_reuses_equal_order_under_its_own_id(example_table_metadata_v2: dict[str, Any]) -> None:
    metadata = _commit_stale_sort_order(
        example_table_metadata_v2, SORT_BY_Y, SORT_BY_X.model_copy(update={"order_id": 5}), stale=SORT_BY_X
    )

    # The equal order is 5, so that is the id reused and made the default
    assert [sort.order_id for sort in metadata.sort_orders] == [3, 4, 5]
    assert metadata.default_sort_order_id == 5


def test_add_sort_order_reuses_equal_order_without_the_highest_id(example_table_metadata_v2: dict[str, Any]) -> None:
    metadata = _commit_stale_sort_order(
        example_table_metadata_v2, SORT_BY_X, SORT_BY_Y.model_copy(update={"order_id": 5}), stale=SORT_BY_X
    )

    # The reused order 4 is not the highest id, and SetDefaultSortOrderUpdate(-1) still makes it the default
    assert [sort.order_id for sort in metadata.sort_orders] == [3, 4, 5]
    assert metadata.default_sort_order_id == 4


def test_add_sort_order_reassigns_taken_id(example_table_metadata_v2: dict[str, Any]) -> None:
    metadata = _commit_stale_sort_order(example_table_metadata_v2, SORT_BY_X, stale=SORT_BY_Y)

    # Id 4 is taken by a different order, so like Java the stale order gets the next id, 5
    assert metadata.sort_orders[-1] == SORT_BY_Y.model_copy(update={"order_id": 5})
    assert metadata.default_sort_order_id == 5


def test_add_sort_order_defaults_to_the_last_added_order_in_a_transaction(example_table_metadata_v2: dict[str, Any]) -> None:
    metadata = _commit_stale_sort_order(example_table_metadata_v2, SORT_BY_Y, stale=(SORT_BY_X, SORT_BY_Y))

    # Both orders are built against metadata that only has order 3, so both come in with id 4. The other writer
    # already added y as 4, so x is listed as 5 and y reuses 4. The default is the order added last in the
    # transaction, 4, like Java's lastAddedOrderId, and not the highest id 5 that is also added here
    assert [sort.order_id for sort in metadata.sort_orders] == [3, 4, 5]
    assert metadata.default_sort_order_id == 4
    assert metadata.sort_order().fields == SORT_BY_Y.fields


def test_add_sort_order_last_added_is_not_the_lowest_id_added(example_table_metadata_v2: dict[str, Any]) -> None:
    metadata = _commit_stale_sort_order(
        example_table_metadata_v2,
        SORT_BY_X,
        stale=(SORT_BY_Y.model_copy(update={"order_id": 5}), SORT_BY_X_DESC.model_copy(update={"order_id": 6})),
    )

    # The other writer added x as 4. Neither order of the stale writer equals an order in the metadata, and ids 5 and
    # 6 are free, so both are listed as they come in. The default is the order added last, 6, and not the lowest id
    # added in the transaction, 5
    assert [sort.order_id for sort in metadata.sort_orders] == [3, 4, 5, 6]
    assert metadata.default_sort_order_id == 6


def test_add_sort_order_last_added_is_not_the_lowest_id_after_a_reuse(example_table_metadata_v2: dict[str, Any]) -> None:
    metadata = _commit_stale_sort_order(
        example_table_metadata_v2,
        SORT_BY_Y.model_copy(update={"order_id": 5}),
        stale=(SORT_BY_X, SORT_BY_Y.model_copy(update={"order_id": 5})),
    )

    # The other writer added y as 5. The stale writer first adds x, which keeps the free id 4 it came in with, unlike
    # Java which recomputes the id, and is appended. It then adds y again, which reuses the id of the equal order, 5,
    # instead of being appended. The default is the order added last, 5, and not the lowest id in the transaction, 4
    assert [sort.order_id for sort in metadata.sort_orders] == [3, 5, 4]
    assert metadata.default_sort_order_id == 5
    assert metadata.sort_order().fields == SORT_BY_Y.fields
