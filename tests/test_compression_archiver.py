"""
Unit tests for CompressionArchiver.
"""
import tempfile
import shutil
from pathlib import Path
import pytest
from CAL.compression import CompressionArchiver, ArchiveEntry


class TestCompressionArchiver:
    """Test CompressionArchiver class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def archiver(self, temp_dir):
        """Create an archiver instance for tests."""
        return CompressionArchiver(session_id="test_session", base_dir=temp_dir)
    
    def test_initialization_with_session_id(self, temp_dir):
        """Test archiver initialization creates correct directory structure."""
        archiver = CompressionArchiver(session_id="test_123", base_dir=temp_dir)
        assert archiver.session_id == "test_123"
        assert archiver.session_dir == Path(temp_dir) / "cal_memory_test_123"
        assert archiver.history_path == archiver.session_dir / "history.md"
    
    def test_initialization_without_base_dir(self):
        """Test archiver uses system temp when base_dir not provided."""
        archiver = CompressionArchiver(session_id="test_session")
        assert archiver._base_dir == Path(tempfile.gettempdir())
        assert archiver.session_dir == Path(tempfile.gettempdir()) / "cal_memory_test_session"
    
    def test_write_context_file_creates_file(self, archiver):
        """Test that write_context_file creates a file."""
        file_path = archiver.write_context_file(
            filename="test_context",
            content="# Test Content\n\nSome content here.",
            message_range="1-10",
            tools_used=["read_file"],
            key_files=["test.py"],
            summary="Test summary",
        )
        
        assert file_path.exists()
        assert file_path.name == "test_context.md"
        assert file_path.read_text() == "# Test Content\n\nSome content here."
    
    def test_write_context_file_updates_history(self, archiver):
        """Test that write_context_file updates history.md."""
        archiver.write_context_file(
            filename="test_context",
            content="Test content",
            message_range="1-10",
            tools_used=["read_file"],
            key_files=["test.py"],
            summary="Test summary",
        )
        
        assert archiver.history_path.exists()
        history_content = archiver.history_path.read_text()
        assert "test_context.md" in history_content
        assert "Test summary" in history_content
        assert "read_file" in history_content
    
    def test_filename_collision_handling(self, archiver):
        """Test that filename collisions are handled with counter."""
        # Create first file
        path1 = archiver.write_context_file(
            filename="test_context",
            content="Content 1",
            message_range="1-5",
            tools_used=[],
            key_files=[],
            summary="First",
        )
        
        # Create second file with same semantic name
        path2 = archiver.write_context_file(
            filename="test_context",
            content="Content 2",
            message_range="6-10",
            tools_used=[],
            key_files=[],
            summary="Second",
        )
        
        assert path1 != path2
        assert path1.name == "test_context.md"
        assert path2.name == "test_context_1.md"
        assert path1.exists()
        assert path2.exists()
    
    def test_make_safe_filename(self, archiver):
        """Test filename sanitization."""
        assert archiver._make_safe_filename("Test File Name") == "test_file_name"
        assert archiver._make_safe_filename("test@file#name") == "testfilename"
        assert archiver._make_safe_filename("") == "context"
        assert len(archiver._make_safe_filename("a" * 100)) <= 50
    
    def test_has_archived_context(self, archiver):
        """Test has_archived_context returns correct state."""
        assert archiver.has_archived_context() is False
        
        archiver.write_context_file(
            filename="test",
            content="Content",
            message_range="1-5",
            tools_used=[],
            key_files=[],
            summary="Test",
        )
        
        assert archiver.has_archived_context() is True
    
    def test_get_history_reference(self, archiver):
        """Test get_history_reference returns correct format."""
        # No archived context
        assert archiver.get_history_reference() == ""
        
        # With archived context
        archiver.write_context_file(
            filename="test_context",
            content="Content",
            message_range="1-5",
            tools_used=["read_file"],
            key_files=["test.py"],
            summary="Test summary",
        )
        
        reference = archiver.get_history_reference()
        assert "[Previous conversation context has been archived to files.]" in reference
        assert "history.md" in reference
        assert "test_context.md" in reference
        assert "Test summary" in reference
    
    def test_cleanup(self, archiver):
        """Test cleanup removes directory and clears entries."""
        archiver.write_context_file(
            filename="test",
            content="Content",
            message_range="1-5",
            tools_used=[],
            key_files=[],
            summary="Test",
        )
        
        assert archiver.has_archived_context() is True
        assert archiver.session_dir.exists()
        
        archiver.cleanup()
        
        assert archiver.has_archived_context() is False
        assert not archiver.session_dir.exists()
    
    def test_to_dict_serialization(self, archiver):
        """Test serialization to dict."""
        archiver.write_context_file(
            filename="test",
            content="Content",
            message_range="1-5",
            tools_used=["read_file"],
            key_files=["test.py"],
            summary="Test summary",
        )
        
        data = archiver.to_dict()
        assert data["session_id"] == "test_session"
        assert len(data["entries"]) == 1
        assert data["entries"][0]["filename"] == "test.md"
        assert data["entries"][0]["summary"] == "Test summary"
    
    def test_from_dict_deserialization(self, temp_dir):
        """Test deserialization from dict."""
        data = {
            "session_id": "restored_session",
            "base_dir": temp_dir,
            "entries": [
                {
                    "filename": "test.md",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message_range": "1-5",
                    "tools_used": ["read_file"],
                    "key_files": ["test.py"],
                    "summary": "Test summary",
                }
            ],
        }
        
        archiver = CompressionArchiver.from_dict(data)
        assert archiver.session_id == "restored_session"
        assert len(archiver._entries) == 1
        assert archiver._entries[0].filename == "test.md"
        assert archiver._entries[0].summary == "Test summary"
