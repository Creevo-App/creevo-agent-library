from CAL.content_blocks import TextBlock
from CAL.message import Message, MessageRole


def test_message_defaults():
    message = Message(role=MessageRole.USER, content=[TextBlock(text="hello")])

    assert message.role == MessageRole.USER
    assert message.content[0].text == "hello"
    assert message.usage == {}
    assert message.metadata == {}


def test_message_repr_includes_fields():
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="hello")],
        usage={"prompt_tokens": 1},
        metadata={"source": "test"},
    )

    assert repr(message) == (
        "Message(role=MessageRole.ASSISTANT, "
        "content=[TextBlock(text='hello')], "
        "usage={'prompt_tokens': 1}, "
        "metadata={'source': 'test'})"
    )
