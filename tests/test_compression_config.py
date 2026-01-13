"""
Unit tests for CompressionConfig.
"""
import pytest
from CAL.compression import CompressionConfig


class TestCompressionConfig:
    """Test CompressionConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are correct."""
        config = CompressionConfig()
        assert config.keep_recent_tokens == 10000
        assert config.max_summary_tokens == 2000
        assert config.preserve_tool_names is True
        assert config.archive_tool_results is True
        assert config.archive_file_reads is True
        assert config.summary_style == "narrative"
        assert config.compression_ratio == 0.3
    
    def test_custom_values(self):
        """Test that all fields can be set."""
        config = CompressionConfig(
            keep_recent_tokens=5000,
            max_summary_tokens=1000,
            preserve_tool_names=False,
            archive_tool_results=False,
            archive_file_reads=False,
            summary_style="bullet",
            compression_ratio=0.5,
        )
        assert config.keep_recent_tokens == 5000
        assert config.max_summary_tokens == 1000
        assert config.preserve_tool_names is False
        assert config.archive_tool_results is False
        assert config.archive_file_reads is False
        assert config.summary_style == "bullet"
        assert config.compression_ratio == 0.5
