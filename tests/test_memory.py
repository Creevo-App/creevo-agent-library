from CAL.content_blocks import TextBlock
from CAL.memory import FullCompressionMemory
from CAL.message import Message, MessageRole


def make_text_message(role: MessageRole, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def test_full_compression_memory_compresses():
    memory = FullCompressionMemory(max_items=4)
    memory.add_message(make_text_message(MessageRole.USER, "first"))
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "second"))
    memory.add_message(make_text_message(MessageRole.USER, "third"))
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "fourth"))
    memory.add_message(make_text_message(MessageRole.USER, "fifth"))

    history = memory.get_history()

    assert len(history) == 4
    assert history[1].metadata.get("compressed") is True
    assert "[Summary of 2 previous turns:]" in history[1].content


def test_full_compression_memory_round_trip_json():
    memory = FullCompressionMemory(max_items=10)
    memory.add_message(make_text_message(MessageRole.USER, "hello"))
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "world"))

    payload = memory.to_json()
    restored = FullCompressionMemory.from_json(payload)

    history = restored.get_history()
    assert len(history) == 2
    assert history[0].role == MessageRole.USER
    assert history[1].role == MessageRole.ASSISTANT
    assert history[0].content[0].text == "hello"
    assert history[1].content[0].text == "world"


def test_full_compression_memory_clone_is_independent():
    memory = FullCompressionMemory(max_items=10)
    memory.add_message(make_text_message(MessageRole.USER, "one"))

    clone = memory.clone()
    clone.add_message(make_text_message(MessageRole.USER, "two"))

    assert len(memory.get_history()) == 1
    assert len(clone.get_history()) == 2
