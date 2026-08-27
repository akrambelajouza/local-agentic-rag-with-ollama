import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from local_rag.ingestion import CorpusDocument, load_documents


class DocumentLoadingTests(unittest.TestCase):
    def test_loads_the_documented_jsonl_corpus(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "sample.jsonl"
            dataset_path.write_text(
                "\n".join(
                    (
                        '{"url":"https://example.com/a","title":"A","raw_text":"Alpha"}',
                        "",
                        '{"url":"https://example.com/b","title":"B","raw_text":"Beta"}',
                    )
                ),
                encoding="utf-8",
            )

            documents = load_documents(dataset_path)

        self.assertEqual(
            documents,
            [
                CorpusDocument("https://example.com/a", "A", "Alpha"),
                CorpusDocument("https://example.com/b", "B", "Beta"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
