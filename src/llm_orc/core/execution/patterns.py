"""Shared patterns for agent instance naming."""

import re

# Matches agent instance names like "researcher[0]", "writer[2]"
INSTANCE_PATTERN = re.compile(r"^(.+)\[(\d+)\]$")
