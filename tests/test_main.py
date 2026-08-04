import src.pdf_data_extractor.main
from src.pdf_data_extractor.schemas import (
    DocumentExtractionResult,
    GenericDocumentData,
)


def test_main_uses_pdf_classification_flow(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        src.pdf_data_extractor.main.sys,
        "argv",
        ["main.py", "data/generic.pdf"],
    )

    captured_path = None

    def fake_classify_pdf_with_groq(pdf_path):
        nonlocal captured_path
        captured_path = pdf_path
        return DocumentExtractionResult(
            document_type="generic",
            data=GenericDocumentData(
                title="Meeting Notes",
            ),
        )

    monkeypatch.setattr(
        src.pdf_data_extractor.main,
        "classify_pdf_with_groq",
        fake_classify_pdf_with_groq,
    )

    src.pdf_data_extractor.main.main()

    output = capsys.readouterr().out

    assert str(captured_path) == "data/generic.pdf"
    assert "Final response:" in output
    assert '"document_type": "generic"' in output
    assert '"title": "Meeting Notes"' in output
