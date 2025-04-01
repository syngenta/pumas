from abc import ABC, abstractmethod
from typing import Callable

import pytest

from pumas.architecture.catalogue import Catalogue


class MockItem:
    pass


class UnrelatedClass:
    pass


class AbstractItem(ABC):
    @abstractmethod
    def do_something(self):
        pass


class ConcreteItem(AbstractItem):
    def do_something(self):
        return "Done"


def test_register_non_item_type():
    catalogue = Catalogue(MockItem)
    with pytest.raises(TypeError):
        catalogue.register("not_an_item", "not_a_mock_item")


def test_remove_item():
    catalogue = Catalogue(MockItem)
    item = MockItem()
    catalogue.register("test_item", item)
    catalogue.remove("test_item")
    assert "test_item" not in catalogue.list_items()


def test_remove_nonexistent_item():
    catalogue = Catalogue(MockItem)
    with pytest.raises(ValueError):
        catalogue.remove("nonexistent")


def test_retrieve_item():
    catalogue = Catalogue(MockItem)
    item = MockItem()
    catalogue.register("test_item", item)
    retrieved_item = catalogue.get("test_item")
    assert retrieved_item == item


def test_retrieve_nonexistent_item():
    catalogue = Catalogue(MockItem)
    with pytest.raises(ValueError):
        catalogue.get("nonexistent")


def test_list_items():
    catalogue = Catalogue(MockItem)
    item1 = MockItem()
    item2 = MockItem()
    catalogue.register("item1", item1)
    catalogue.register("item2", item2)
    assert set(catalogue.list_items()) == {"item1", "item2"}


def test_decorator_registration():
    catalogue = Catalogue(MockItem)

    @catalogue.register_decorator("decorated_item")
    class DecoratedItem(MockItem):
        pass

    _ = DecoratedItem()

    assert "decorated_item" in catalogue.list_items()


def test_repeated_registration():
    catalogue = Catalogue(MockItem)
    item = MockItem()
    catalogue.register("test_item", item)
    with pytest.raises(ValueError):
        catalogue.register("test_item", item)


def test_register_non_subclass():
    catalogue = Catalogue(MockItem)
    with pytest.raises(TypeError) as exc_info:

        @catalogue.register_decorator("non_subclass_item")
        class NonSubclassItem(UnrelatedClass):
            pass

        _ = NonSubclassItem()

    assert "Provided class is not a subclass of MockItem" in str(exc_info.value)


def test_register_callable():
    catalogue = Catalogue(Callable)

    def example_function():
        return "Example"

    catalogue.register("example_function", example_function)
    retrieved_function = catalogue.get("example_function")
    assert retrieved_function() == "Example"


def test_register_concrete_class_of_abstract_base():
    catalogue = Catalogue(AbstractItem)
    concrete_item = ConcreteItem()
    catalogue.register("concrete_item", concrete_item)
    retrieved_item = catalogue.get("concrete_item")
    assert retrieved_item.do_something() == "Done"
