"""Relocation shim — Plexus adapter moved to :mod:`llm_orc.core.session.plexus_adapter`.

Cycle-8 WP-B8: the surviving ADR-009/010 optional KG substrate adapter
relocated out of the ``agentic/`` deletion target. Deleted with the
package at WP-F8.
"""

from llm_orc.core.session.plexus_adapter import PlexusAdapter as PlexusAdapter
