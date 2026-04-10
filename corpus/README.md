# Corpus Directory

Place documents here for ingestion. Supported formats:

- `.txt` — plain text files
- `.pdf` — PDF documents (text-based, not scanned)
- `.docx` — Microsoft Word documents

Run `python main.py ingest` to process all documents.

If this directory is empty, `ingest` will automatically load a sample
from the SQuAD validation set for development and evaluation purposes.
