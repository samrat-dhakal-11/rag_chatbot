"""
pdf_parser.py
Uses LlamaParse to turn an uploaded PDF into clean markdown/text.
"""

import tempfile
from llama_parse import LlamaParse
from config import LLAMA_PARSE_KEY

_parser = LlamaParse(
    api_key=LLAMA_PARSE_KEY,
    result_type="markdown",   # "markdown" keeps tables/headings; use "text" for plain text
    verbose=True,
)


def parse_pdf(uploaded_file) -> str:
    """
    uploaded_file: a Streamlit UploadedFile (from st.file_uploader)
    returns: parsed text (str) for that PDF
    """
    # LlamaParse needs a real file path, so we write the upload to a temp file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    documents = _parser.load_data(tmp_path)
    full_text = "\n\n".join(doc.text for doc in documents)
    return full_text