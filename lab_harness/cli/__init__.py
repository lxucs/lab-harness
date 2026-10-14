"""Subscribers to the agent's event stream. The runtime never imports from here."""

from lab_harness.cli.json_sink import json_sink

__all__ = ["json_sink"]
