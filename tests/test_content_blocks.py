import base64

from CAL.content_blocks import (
    ImageBlock,
    ImageSource,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def test_content_block_reprs():
    text = TextBlock(text="hello")
    image = ImageBlock(source=ImageSource(type="base64", media_type="image/png", data=b"abc"))
    tool_use = ToolUseBlock(id="tool-use-1", name="do_work", input={"count": 2})
    tool_result = ToolResultBlock(tool_use_id="tool-use-2", content="done", name="tool")

    assert repr(text) == "TextBlock(text='hello')"
    assert repr(image) == "ImageBlock(source=ImageSource(type=base64, media_type=image/png))"
    assert repr(tool_use) == "ToolUseBlock(id=tool-use-1, name=do_work, input={'count': 2}, thought=None)"
    assert repr(tool_result) == "ToolResultBlock(tool_use_id=tool-use-2, is_error=False, name=tool)"


def test_text_block_round_trip():
    block = TextBlock(text="hello")
    data = block.to_dict()

    assert data == {"type": "text", "text": "hello"}

    restored = TextBlock.from_dict(data)
    assert repr(restored) == repr(block)


def test_image_block_round_trip_bytes():
    source = ImageSource(type="base64", media_type="image/png", data=b"abc")
    block = ImageBlock(source=source)
    data = block.to_dict()

    assert data["source"]["type"] == "base64"
    assert data["source"]["media_type"] == "image/png"
    assert isinstance(data["source"]["data"], str)

    restored = ImageBlock.from_dict(data)
    assert restored.source.type == "base64"
    assert restored.source.media_type == "image/png"
    assert isinstance(restored.source.data, (bytes, bytearray))
    assert restored.source.data == b"abc"
    assert repr(restored) == repr(block)


def test_tool_use_block_round_trip():
    block = ToolUseBlock(
        id="tool-use-1",
        name="do_work",
        input={"count": 2},
        thought="plan",
        thought_signature=b"sig",
    )
    data = block.to_dict()

    assert data["thought_signature"] == base64.b64encode(b"sig").decode("utf-8")

    restored = ToolUseBlock.from_dict(data)
    assert restored.thought_signature == b"sig"
    assert repr(restored) == repr(block)


def test_tool_result_block_round_trip_list_content():
    block = ToolResultBlock(
        tool_use_id="tool-use-2",
        content=[TextBlock(text="done")],
        is_error=True,
        name="tool",
        metadata={"status": "ok"},
    )
    data = block.to_dict()

    restored = ToolResultBlock.from_dict(data)
    assert repr(restored) == repr(block)


def test_tool_result_block_round_trip_string_content():
    block = ToolResultBlock(
        tool_use_id="tool-use-3",
        content="done",
        is_error=False,
        name="tool",
    )
    data = block.to_dict()

    restored = ToolResultBlock.from_dict(data)
    assert repr(restored) == repr(block)


def test_tool_result_gemini_content_form_includes_error():
    block = ToolResultBlock(
        tool_use_id="tool-use-4",
        content="oops",
        is_error=True,
        name="tool",
    )
    parts = block.gemini_content_form()

    assert isinstance(parts, list)
    response = parts[0].function_response.response
    assert response["result"] == "oops"
    assert response["error"] is True


def test_tool_result_gemini_content_form_list_content():
    block = ToolResultBlock(
        tool_use_id="tool-use-5",
        content=[TextBlock(text="ok")],
        name="tool",
    )
    parts = block.gemini_content_form()

    response = parts[0].function_response.response
    assert response["result"] == ["ok"]
