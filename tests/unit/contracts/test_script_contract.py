"""Unit tests for ScriptContract base class and contract system."""

import pytest

from llm_orc.contracts.script_contract import ScriptContract


class TestScriptContract:
    """Test the core ScriptContract abstract interface."""

    def test_script_contract_is_abstract_base_class(self) -> None:
        """Test that ScriptContract cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ScriptContract()  # type: ignore

    def test_script_contract_defines_required_abstract_methods(self) -> None:
        """Test that ScriptContract defines all required abstract methods."""
        abstract_methods = ScriptContract.__abstractmethods__

        expected_methods = {
            "metadata",
            "input_schema",
            "output_schema",
            "execute",
            "get_test_cases",
        }

        assert abstract_methods == expected_methods

    def test_script_contract_subclass_must_implement_all_methods(self) -> None:
        """Test that incomplete ScriptContract subclass raises TypeError."""

        class IncompleteScript(ScriptContract):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteScript()  # type: ignore
