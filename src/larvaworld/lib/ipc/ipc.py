#   Copyright 2017 Dan Krause
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from __future__ import annotations

from typing import Any, Callable, TypeVar
import json
import socket
import socketserver
import struct

try:
    from socketserver import ThreadingUnixStreamServer as StreamServer
except ImportError:
    from socketserver import ThreadingTCPServer as StreamServer
from socketserver import ThreadingTCPServer

__all__: list[str] = [
    "Message",
    "Client",
    "Server",
]

__displayname__ = "Client-Server remote messaging"


class IPCError(Exception):
    pass


class UnknownMessageClass(IPCError):
    pass


class InvalidSerialization(IPCError):
    pass


class ConnectionClosed(IPCError):
    pass


def _read_objects(sock: socket.socket) -> list["Message"]:
    """
    Reads and deserializes a message object from a socket.

    This function reads a 4-byte header from the socket to determine the size of the incoming message.
    It then reads the message data from the socket, deserializes it from JSON, and returns the resulting Message object.

    Args:
        sock (socket.socket): The socket from which to read the message.

    Returns:
        list[Message]: The deserialized message objects.

    Raises:
        ConnectionClosed: If the connection is closed or if no data is received.
    """
    header = sock.recv(4)
    if len(header) == 0:
        raise ConnectionClosed()
    size = struct.unpack("!i", header)[0]
    data = sock.recv(size - 4)
    if len(data) == 0:
        raise ConnectionClosed()
    return Message.deserialize(json.loads(data.decode()))


def _write_objects(sock: socket.socket, objects: list["Message"]) -> None:
    """
    Serializes a list of objects and sends them over a socket.

    Args:
        sock (socket.socket): The socket through which the data will be sent.
        objects (list): A list of objects that have a `serialize` method, which returns a JSON-serializable representation of the object.
    """
    data = json.dumps([o.serialize() for o in objects])
    sock.sendall(struct.pack("!i", len(data) + 4))
    sock.sendall(str.encode(data))


T = TypeVar("T", bound="Message")


def _recursive_subclasses(cls: type[T]) -> dict[str, type[T]]:
    """
    Recursively finds all subclasses of a given class and returns a dictionary mapping
    subclass names to subclass objects.

    Args:
        cls (type): The class for which to find all subclasses.

    Returns:
        dict: A dictionary where the keys are subclass names (str) and the values are
              the subclass objects (type).
    """
    classmap: dict[str, type[T]] = {}
    for subcls in cls.__subclasses__():
        classmap[subcls.__name__] = subcls
        classmap.update(_recursive_subclasses(subcls))
    return classmap


class Message:
    """
    A base class for serializable messages.
    """

    @classmethod
    def deserialize(cls: type[T], objects: list[Any]) -> list[T]:
        """
        Deserialize a list of objects into their corresponding class instances.

        Args:
            cls (type): The base class to use for deserialization.
            objects (list): A list of serialized objects, where each object is either
                            an instance of `Message` or a dictionary containing the
                            keys "class", "args", and "kwargs".

        Returns:
            list: A list of deserialized objects, where each object is an instance
                  of the class specified in the serialized data.

        Raises:
            UnknownMessageClass: If the class specified in the serialized data is not found.
            InvalidSerialization: If there is an error in the deserialization process.
        """
        classmap = _recursive_subclasses(cls)
        serialized: list[T] = []
        for obj in objects:
            if isinstance(obj, Message):
                serialized.append(obj)
            else:
                try:
                    serialized.append(
                        classmap[obj["class"]](*obj["args"], **obj["kwargs"])  # type: ignore[misc]
                    )
                except KeyError as e:
                    raise UnknownMessageClass(e)
                except TypeError as e:
                    raise InvalidSerialization(e)
        return serialized

    def serialize(self) -> dict[str, Any]:
        """
        Serializes the current object instance into a dictionary.

        The dictionary contains the class name, positional arguments, and keyword arguments
        required to recreate the object instance.

        Returns:
            dict: A dictionary with the following keys:
                - "class" (str): The name of the class of the current object instance.
                - "args" (list): The positional arguments used to initialize the object.
                - "kwargs" (dict): The keyword arguments used to initialize the object.
        """
        args, kwargs = self._get_args()
        return {"class": type(self).__name__, "args": args, "kwargs": kwargs}

    def _get_args(self) -> tuple[list[Any], dict[str, Any]]:
        return [], {}

    def __repr__(self) -> str:
        """
        Return a string representation of the object for debugging.

        This method serializes the object and constructs a string that represents
        the object in a way that is useful for debugging. The string includes the
        class name, positional arguments, and keyword arguments.

        Returns:
            str: A string representation of the object.
        """
        r = self.serialize()
        args = ", ".join([repr(arg) for arg in r["args"]])
        kwargs = "".join([f", {k}={v!r}" for k, v in r["kwargs"].items()])
        name = r["class"]
        return f"{name}({args}{kwargs})"


class Client:
    """
    A client class for handling IPC (Inter-Process Communication) with a server.
    """

    def __init__(self, server_address: tuple[str, int] | str | bytes) -> None:
        self.addr: tuple[str, int] | str | bytes = server_address
        # print(self.addr)
        # raise
        if isinstance(self.addr, (str, bytes)):
            address_family = socket.AF_UNIX
        else:
            address_family = socket.AF_INET
        self.sock = socket.socket(address_family, socket.SOCK_STREAM)

    def connect(self) -> None:
        self.sock.connect(self.addr)

    def close(self) -> None:
        self.sock.close()

    def __enter__(self) -> "Client":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def send(self, objects: list[Message]) -> list[Message]:
        _write_objects(self.sock, objects)
        return _read_objects(self.sock)


class Server(StreamServer):
    """
    A server class that handles inter-process communication (IPC) using a stream-based server.
    """

    def __init__(
        self,
        server_address: tuple[str, int] | str | bytes,
        callback: Callable[[list[Message]], list[Message]] | None,
        bind_and_activate: bool = True,
    ) -> None:
        if not callable(callback):
            callback = lambda x: []

        class IPCHandler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                while True:
                    try:
                        results = _read_objects(self.request)
                    except ConnectionClosed:
                        return
                    _write_objects(self.request, callback(results))

        if (
            ThreadingTCPServer in type(self).mro()
        ):  # specifically an AF_INET server in this case
            self.address_family = socket.AF_INET
        elif isinstance(server_address, (str, bytes)):
            self.address_family = socket.AF_UNIX
        else:
            self.address_family = socket.AF_INET

        socketserver.TCPServer.__init__(
            self, server_address, IPCHandler, bind_and_activate
        )
