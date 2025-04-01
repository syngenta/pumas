import base64

from IPython.display import Image, display


def mm_ink(graph_bytes):
    """Given a bytes object holding a Mermaid-format graph,
    return a URL that will generate the image."""
    base64_bytes = base64.b64encode(graph_bytes)
    base64_string = base64_bytes.decode("ascii")
    return "https://mermaid.ink/img/" + base64_string


def mm_display(graph_bytes):
    """Given a bytes object holding a Mermaid-format graph, display it."""
    display(Image(url=mm_ink(graph_bytes)))


def mm(graph):
    """Given a string containing a Mermaid-format graph, display it."""
    graph_bytes = graph.encode("ascii")
    mm_display(graph_bytes)


def mm_link(graph):
    """Given a string containing a Mermaid-format graph, return URL for display."""
    graph_bytes = graph.encode("ascii")
    return mm_ink(graph_bytes)


def mm_path(path):
    """Given a path to a file containing a Mermaid-format graph, display it"""
    with open(path, "rb") as f:
        graph_bytes = f.read()
    mm_display(graph_bytes)


Mermaid = mm
