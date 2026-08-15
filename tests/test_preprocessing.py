"""Unit tests for src/utils/preprocessing.py."""

import base64
import io

import pytest
from PIL import Image

from src.utils.preprocessing import (
    ImagePreprocessor,
    ModalityDetector,
    PreprocessingError,
    TextPreprocessor,
)


def make_png_bytes(width: int = 8, height: int = 8) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 30, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def make_png_data_url(width: int = 8, height: int = 8) -> str:
    encoded = base64.b64encode(make_png_bytes(width, height)).decode()
    return f"data:image/png;base64,{encoded}"


class TestTextPreprocessor:
    @pytest.fixture
    def processor(self) -> TextPreprocessor:
        return TextPreprocessor()

    def test_clean_text_collapses_whitespace_and_strips(self, processor):
        assert processor.clean_text("  hello \n\t world  ") == "hello world"

    def test_clean_text_coerces_non_strings(self, processor):
        assert processor.clean_text(42) == "42"

    def test_clean_text_removes_urls_only_when_requested(self, processor):
        text = "see https://example.com/docs now"
        assert processor.clean_text(text) == text
        assert processor.clean_text(text, remove_urls=True) == "see now"

    def test_clean_text_removes_emails_only_when_requested(self, processor):
        text = "mail me at user.name@example.com please"
        assert processor.clean_text(text) == text
        assert processor.clean_text(text, remove_emails=True) == "mail me at please"

    def test_truncate_text_returns_input_when_short_enough(self, processor):
        assert processor.truncate_text("short", 10) == "short"

    def test_truncate_text_truncate_strategy(self, processor):
        assert processor.truncate_text("abcdefghij", 4) == "abcd"

    def test_truncate_text_middle_strategy_keeps_both_ends(self, processor):
        result = processor.truncate_text("abcdefghij", 4)
        assert result == "abcd"
        middle = processor.truncate_text("abcdefghij", 4, strategy="middle")
        assert middle == "ab ... ij"

    def test_truncate_text_sentences_strategy_keeps_whole_sentences(self, processor):
        text = "One two. Three four. Five six."
        result = processor.truncate_text(text, 22, strategy="sentences")
        assert result == "One two. Three four."

    def test_truncate_text_rejects_unknown_strategy(self, processor):
        with pytest.raises(ValueError, match="Unknown truncation strategy"):
            processor.truncate_text("abcdefghij", 2, strategy="nope")

    def test_validate_text_accepts_valid_text(self, processor):
        assert processor.validate_text("hello", max_length=10) is True

    @pytest.mark.parametrize(
        "text, message",
        [
            (123, "must be a string"),
            ("   ", "cannot be empty"),
        ],
    )
    def test_validate_text_rejects_invalid_text(self, processor, text, message):
        with pytest.raises(PreprocessingError, match=message):
            processor.validate_text(text)

    def test_validate_text_enforces_max_length(self, processor):
        with pytest.raises(PreprocessingError, match="exceeds maximum"):
            processor.validate_text("abcdef", max_length=3)


class TestImagePreprocessor:
    @pytest.fixture
    def processor(self) -> ImagePreprocessor:
        return ImagePreprocessor()

    def test_parse_data_url_returns_format_and_bytes(self, processor):
        image_format, image_bytes = processor.parse_data_url(make_png_data_url())
        assert image_format == "png"
        assert image_bytes.startswith(b"\x89PNG")

    def test_parse_data_url_normalizes_format_case(self, processor):
        encoded = base64.b64encode(make_png_bytes()).decode()
        image_format, _ = processor.parse_data_url(f"data:image/PNG;base64,{encoded}")
        assert image_format == "png"

    def test_parse_data_url_rejects_non_data_url(self, processor):
        with pytest.raises(PreprocessingError, match="Invalid data URL format"):
            processor.parse_data_url("https://example.com/cat.png")

    def test_parse_data_url_rejects_unsupported_format(self, processor):
        encoded = base64.b64encode(b"not-an-image").decode()
        with pytest.raises(PreprocessingError, match="Unsupported image format"):
            processor.parse_data_url(f"data:image/tiff;base64,{encoded}")

    def test_parse_data_url_rejects_invalid_base64(self, processor):
        with pytest.raises(PreprocessingError, match="Invalid base64 encoding"):
            processor.parse_data_url("data:image/png;base64,@@@not-base64@@@")

    def test_parse_data_url_enforces_max_file_size(self, processor):
        processor.max_file_size = 10
        with pytest.raises(PreprocessingError, match="exceeds maximum"):
            processor.parse_data_url(make_png_data_url())

    def test_validate_image_accepts_data_url(self, processor):
        image_format, image_bytes = processor.validate_image(make_png_data_url())
        assert image_format == "png"
        assert len(image_bytes) > 0

    def test_validate_image_detects_format_from_bytes(self, processor):
        image_format, image_bytes = processor.validate_image(make_png_bytes())
        assert image_format == "png"
        assert image_bytes.startswith(b"\x89PNG")

    def test_validate_image_rejects_wrong_type(self, processor):
        with pytest.raises(PreprocessingError, match="must be string .* or bytes"):
            processor.validate_image(123)

    def test_validate_image_rejects_corrupt_payload(self, processor):
        encoded = base64.b64encode(b"\x89PNG-but-not-really").decode()
        with pytest.raises(PreprocessingError, match="Invalid image data"):
            processor.validate_image(f"data:image/png;base64,{encoded}")

    def test_validate_image_enforces_max_dimension(self, processor):
        processor.max_dimension = 4
        with pytest.raises(PreprocessingError, match="exceed maximum"):
            processor.validate_image(make_png_data_url(8, 8))

    def test_validate_image_enforces_max_decoded_size(self, processor):
        processor.max_decoded_size = 10
        with pytest.raises(PreprocessingError, match="Decoded image size too large"):
            processor.validate_image(make_png_data_url(8, 8))

    def test_validate_image_warns_on_format_mismatch(self, processor, caplog):
        encoded = base64.b64encode(make_png_bytes()).decode()
        with caplog.at_level("WARNING"):
            image_format, _ = processor.validate_image(
                f"data:image/gif;base64,{encoded}"
            )
        assert image_format == "gif"
        assert "Format mismatch" in caplog.text

    @pytest.mark.parametrize(
        "payload, expected",
        [
            (b"\xff\xd8\xff\xe0rest", "jpeg"),
            (b"\x89PNG\r\n", "png"),
            (b"RIFF1234WEBPmore", "webp"),
            (b"BMheader", "bmp"),
            (b"GIF89a", "gif"),
            (b"unknown-magic", "unknown"),
        ],
    )
    def test_detect_image_format(self, processor, payload, expected):
        assert processor._detect_image_format(payload) == expected

    def test_extract_image_from_tag_returns_data_url(self, processor):
        data_url = make_png_data_url()
        tag = f'<p>text</p><IMG alt="x" src="{data_url}" />'
        assert processor.extract_image_from_tag(tag) == data_url

    def test_extract_image_from_tag_ignores_remote_sources(self, processor):
        assert processor.extract_image_from_tag('<img src="https://x/y.png">') is None

    def test_extract_image_from_tag_returns_none_without_tag(self, processor):
        assert processor.extract_image_from_tag("plain text") is None


class TestModalityDetector:
    @pytest.fixture
    def detector(self) -> ModalityDetector:
        return ModalityDetector()

    @pytest.mark.parametrize(
        "payload, expected",
        [
            ({"text": "hi"}, "text"),
            ({"image": "data:image/png;base64,AAAA"}, "image"),
            ({"text": "hi", "image": "data:image/png;base64,AAAA"}, "text_image"),
        ],
    )
    def test_detect_modality_for_dict_inputs(self, detector, payload, expected):
        assert detector.detect_modality(payload) == expected

    def test_detect_modality_rejects_empty_dict(self, detector):
        with pytest.raises(PreprocessingError, match="must contain 'text' or 'image'"):
            detector.detect_modality({"other": "value"})

    def test_detect_modality_for_plain_text(self, detector):
        assert detector.detect_modality("  just words  ") == "text"

    def test_detect_modality_for_data_url(self, detector):
        assert detector.detect_modality(make_png_data_url()) == "image"

    def test_detect_modality_for_embedded_img_tag(self, detector):
        payload = f'caption <img src="{make_png_data_url()}">'
        assert detector.detect_modality(payload) == "text_image"

    def test_detect_modality_rejects_unsupported_type(self, detector):
        with pytest.raises(PreprocessingError, match="must be string or dictionary"):
            detector.detect_modality(42)

    def test_extract_content_from_dict(self, detector):
        content = detector.extract_content({"text": "hi", "image": "img"}, "text_image")
        assert content == {"modality": "text_image", "text": "hi", "image": "img"}

    def test_extract_content_from_text(self, detector):
        assert detector.extract_content("hello", "text") == {
            "modality": "text",
            "text": "hello",
        }

    def test_extract_content_from_image(self, detector):
        data_url = make_png_data_url()
        assert detector.extract_content(data_url, "image") == {
            "modality": "image",
            "image": data_url,
        }

    def test_extract_content_splits_mixed_content(self, detector):
        data_url = make_png_data_url()
        content = detector.extract_content(
            f'caption <img src="{data_url}">', "text_image"
        )
        assert content == {
            "modality": "text_image",
            "text": "caption",
            "image": data_url,
        }

    def test_extract_content_falls_back_to_text_without_image_tag(self, detector):
        content = detector.extract_content("no image here", "text_image")
        assert content == {"modality": "text", "text": "no image here"}
