import main


def test_main_uses_pdf_classification_flow(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        main.sys,
        "argv",
        ["main.py", "data/generic.pdf"],
    )

    captured_path = None

    def fake_classify_pdf_with_groq(pdf_path):
        nonlocal captured_path
        captured_path = pdf_path
        return '{"document_type":"generic"}'

    monkeypatch.setattr(
        main,
        "classify_pdf_with_groq",
        fake_classify_pdf_with_groq,
    )

    main.main()

    output = capsys.readouterr().out

    assert str(captured_path) == "data/generic.pdf"
    assert "Final response:" in output
    assert '{"document_type":"generic"}' in output
