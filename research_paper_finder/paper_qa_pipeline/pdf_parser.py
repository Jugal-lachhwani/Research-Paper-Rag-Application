from pathlib import Path


class PdfParser:

    def parse(self, pdf_path: str | Path) -> str:

        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF parsing needs pypdf. Install it with `pip install pypdf`."
            ) from exc

        reader = PdfReader(str(pdf_path))
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n\n".join(pages)
