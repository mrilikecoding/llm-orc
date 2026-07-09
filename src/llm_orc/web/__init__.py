"""Web UI module for llm-orc ensemble management.

Deliberately import-light: importing this package must not build the
FastAPI app (``llm_orc.web.server.create_app`` is imported from its
module directly). The endpoint module imports the ``agentic`` layer
until WP-F8, so an eager app import here would make any
``agentic`` → ``web.serving`` contract import circular.
"""
