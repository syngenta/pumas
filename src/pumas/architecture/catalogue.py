"""
This module provides a `Catalogue` class, which serves as a versatile
registry for different types of items.
It's akin to a factory in the sense that it allows for the creation
and management of various objects. However,
unlike a standard factory pattern that encapsulates object creation,
`Catalogue` focuses more on object registration
and retrieval, aligning with the strategy pattern where it helps in
maintaining a collection of strategies (or items)
for dynamic usage.

The `Catalogue` class differs from a standard factory by enabling
registration and de-registration of items at runtime,
thus offering more flexibility. This feature is particularly useful in
scenarios where the behavior of a system needs
to be extended or modified without altering the existing code structure.

This utility class ensures consistent, factory/strategy-like behavior
across multiple parts of the codebase,
facilitating easier management and scalability of the system.

Simple Usage Examples (Doctest):
--------------------------------
>>> from pumas.architecture.catalogue import Catalogue

>>> class MyItem:
...     pass

>>> # Registering am instance of a class
>>> catalogue = Catalogue(MyItem)
>>> catalogue.register("item1", MyItem())
>>> "item1" in catalogue.list_items()
True

>>> # Using the decorator for registration
>>> @catalogue.register_decorator("decorated_item")
... class DecoratedItem(MyItem):
...     pass
>>> "decorated_item" in catalogue.list_items()
True

>>> # Registering a callable (function) as an item
>>> from typing import Callable
>>> def my_function():
...     return "Functionality"
>>> catalogue_callable = Catalogue(Callable)
>>> catalogue_callable.register("my_function", my_function)
>>> callable(catalogue_callable.get("my_function"))
True
>>> catalogue_callable.get("my_function")()
'Functionality'

>>> # Registering a concrete class of an abstract base class
>>> from abc import ABC, abstractmethod
>>> class AbstractItem(ABC):
...     @abstractmethod
...     def do_something(self):
...         pass
>>> class ConcreteItem(AbstractItem):
...     def do_something(self):
...         return "Done"
>>> catalogue_abc = Catalogue(AbstractItem)
>>> concrete_item = ConcreteItem()
>>> catalogue_abc.register("concrete_item", concrete_item)
>>> catalogue_abc.get("concrete_item").do_something()
'Done'
"""

import inspect
from typing import Any, List


class Catalogue:
    """
    A class for managing items in a factory-like style.

    This class allows for the registration, de-registration, and retrieval of items.
    It's designed to handle items of a specified type,
    ensuring type safety during registration.

    Attributes:
        _items (dict): A private dictionary to store registered items.
        _item_type (type): The type of items this catalogue is designed to handle.

    Methods:
        register(name: str, item): Register a new item.
        remove(name: str): Remove an item from the catalogue.
        get(name: str): Retrieve an item by its name.
        list_items() -> list: List all registered item names.
        register_decorator(name: str): Decorator for registering an item.
    """

    def __init__(self, item_type):
        """
        Initializes an instance of the Catalogue class.

        Args:
            item_type (type): The type of items this catalogue is designed to handle.
        """
        self._items = {}
        self._item_type = item_type

    def register(self, name: str, item: Any) -> None:
        """
        Registers a new item with the given name in the catalogue.
        If the item is a class, it checks if it's a
        subclass of the specified item type.
        If it's an instance, it checks if it's an instance of the item type.

        Args:
            name (str): The name to register the item under.
            item: The item to be registered. Can be a class or an instance.

        Raises:
            TypeError: If the item is neither an instance nor a
                subclass of the specified item type.
            ValueError: If an item with the given name already exists in the catalogue.
        """

        if inspect.isclass(item):
            # Check if it's a subclass of the item type
            if not issubclass(item, self._item_type):
                raise TypeError(
                    f"Provided class is not a subclass of {self._item_type.__name__}."
                )
        else:
            # Check if it's an instance of the item type
            if not isinstance(item, self._item_type):
                raise TypeError(
                    f"Provided object is not an instance of {self._item_type.__name__}."
                )

        if name in self._items:
            raise ValueError(f"Item '{name}' already exists.")

        self._items[name] = item

    def remove(self, name: str) -> None:
        """
        Removes an item from the catalogue by its name.

        Args:
            name (str): The name of the item to be removed.

        Raises:
            ValueError: If no item with the given name exists in the catalogue.
        """
        if name not in self._items:
            raise ValueError(f"Item '{name}' does not exist.")
        del self._items[name]

    def get(self, name: str) -> Any:
        """
        Retrieves an item by its name from the catalogue.

        Args:
            name (str): The name of the item to retrieve.

        Returns:
            The item associated with the given name.

        Raises:
            ValueError: If no item with the given name exists in the catalogue.
        """
        if name not in self._items:
            raise ValueError(f"Item '{name}' does not exist.")
        return self._items.get(name)

    def list_items(self) -> List[Any]:
        """
        Lists all registered item names in the catalogue.

        Returns:
            list: A list of all registered item names.
        """
        return list(self._items.keys())

    def register_decorator(self, name: str) -> Any:
        """
        Provides a decorator for registering items in the catalogue.
        It facilitates the registration of classes
        directly without calling the register method explicitly.

        Args:
            name (str): The name to register the item under.

        Returns:
            A decorator function that takes a class or instance
            and registers it under the provided name.
        """

        def decorator(item):
            self.register(name, item)
            return item

        return decorator
