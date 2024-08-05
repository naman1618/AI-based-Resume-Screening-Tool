from pypdf import PdfReader
import mammoth
from striprtf.striprtf import rtf_to_text
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from functools import cache
from doc2docx import convert
import zipfile
import json
import hashlib
import os
from util.extracted_document import ExtractedDocument
from util.file_extension import FileExtension, get_file_ext
import pathlib
from util.console import console
import time


class TextExtractor:
    def _print(self, output: str):
        print(f"[cyan][TextExtractor][/cyan] {output}")

    def __init__(self, use_linux_support=False, persist_file="cache.json"):
        try:
            with open(persist_file, "r") as f:
                self._persisted_disk_cache = json.loads(f.read())
                self._print(
                    f"Found an existing cache file which has been loaded.\n\tLocation: [green]{pathlib.Path(persist_file).resolve()}[/green]"
                )
        except:
            self._print(
                f"[red]A cache file could not be found.[/red] If you wish to persist data to minimize resource usage, consider calling [bold yellow].persist()[/] method on your extractor instance after loading all documents."
            )
            self._persisted_disk_cache = {}
        self._cache_file = persist_file
        self._use_linux_support = use_linux_support
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Using the following resume/cv, find the creator's name. Return only the candidate's name. If there is no name, just say "None".

Example:
Input:
Calvin Oosse\nChicago, IL 60611 (312) 731-8447\noossec@gmail.com linkedin.com/in/calvin-oosse-a3b2361/\nIT EXECUTIVE: CIO | SVP | VP\nResults-oriented, strategy-driven, people-centric digital technology executive with extensive experience in the \nmost demanding professional services firms working with Boards and C-level executives.

Output:
Calvin Oosse""",
                    ),
                    ("user", "File name: {file_name}\nFile content:\n{content}"),
                ]
            )
            | OllamaLLM(model="llama3.1", temperature=0)
        )

    def persist(self):
        with open(self._cache_file, "w") as f:
            f.write(json.dumps(self._persisted_disk_cache))
            self._print(
                f"Successfully created a cache file at [green]{pathlib.Path(self._cache_file).resolve()}[/green]"
            )

    def _get_author(self, content: str, file_path: str) -> str:
        # Limit to 200 characters since we expect that the name of the candidate is within 200 characters.
        return self._chain.invoke(
            {
                "content": content[:200],
                "file_name": file_path[file_path.rindex("/") + 1 :],
            }
        )

    def _add_to_disk_cache(self, document: ExtractedDocument):
        key = hashlib.sha256(document.content.encode()).hexdigest()
        self._persisted_disk_cache[key] = document.json()

    def _get_persisted_document(self, raw_text: str):
        key = hashlib.sha256(raw_text.encode()).hexdigest()
        return (
            ExtractedDocument.model_validate_json(self._persisted_disk_cache[key])
            if key in self._persisted_disk_cache
            else None
        )

    def extract_pdf(self, file_path: str) -> list[ExtractedDocument]:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            raw_text = "\n".join([x.extract_text() for x in reader.pages])
            persisted = self._get_persisted_document(raw_text)
            if persisted is not None:
                self._print(f"[bold green]FOUND[/] - [magenta underline]{file_path}[/]")
                return [persisted]

            start_time = time.perf_counter()
            self._print(
                f"[bold orange]PROCESSING[/] - [magenta underline]{file_path}[/] ..."
            )
            document = ExtractedDocument(
                content=raw_text,
                file_path=file_path,
                candidate_name=self._get_author(raw_text, file_path),
                file_ext=FileExtension.PDF,
            )
            self._add_to_disk_cache(document)

            self._print(
                f"[bold bright_green]COMPLETE[/] - [magenta underline]{file_path}[/] completed in [bright_green]{(time.perf_counter() - start_time):0.2f} seconds[/bright_green]"
            )
            return [document]

    def extract_docx(self, file_path: str) -> list[ExtractedDocument]:
        with open(file_path, "rb") as f:
            try:
                raw_text = mammoth.extract_raw_text(f).value
                persisted = self._get_persisted_document(raw_text)
                if persisted is not None:
                    self._print(
                        f"[bold green]FOUND[/] - [magenta underline]{file_path}[/]"
                    )
                    return [persisted]

                start_time = time.perf_counter()

                self._print(
                    f"[bold orange]PROCESSING[/] - [magenta underline]{file_path}[/] ..."
                )
                document = ExtractedDocument(
                    content=raw_text,
                    file_path=file_path,
                    candidate_name=self._get_author(raw_text, file_path),
                    file_ext=FileExtension.DOCX,
                )
                self._add_to_disk_cache(document)

                self._print(
                    f"[bold bright_green]COMPLETE[/] - [magenta underline]{file_path}[/] completed in [bright_green]{(time.perf_counter() - start_time):0.2f} seconds[/bright_green]"
                )
                return [document]
            except Exception as e:
                self._print(
                    f"[bold bright_red]ERROR[/] - Could not read [magenta underline]{file_path}[/]. [bright_yellow]This file will be skipped and not processed.[/]\n\n{e}\n"
                )
                return []

    def extract_doc(self, file_path: str) -> list[ExtractedDocument]:
        docx_path = file_path + "x"
        if self._use_linux_support:
            self._print(
                f"[bold bright_yellow]UNSUPPORTED[/] - [magenta underline]{file_path}[/] requires the program to be executed on Windows. [bright_yellow]This file will be skipped and not processed.[/]"
            )
            return []
        if not os.path.exists(docx_path):
            convert(file_path, file_path + "x")
        return self.extract_docx(docx_path)

    def extract_txt(self, file_path: str) -> list[ExtractedDocument]:
        with open(file_path, "r") as f:
            raw_text = f.read()
            persisted = self._get_persisted_document(raw_text)
            if persisted is not None:
                self._print(f"[bold green]FOUND[/] - [magenta underline]{file_path}[/]")
                return [persisted]

            start_time = time.perf_counter()

            self._print(
                f"[bold orange]PROCESSING[/] - [magenta underline]{file_path}[/] ..."
            )
            document = ExtractedDocument(
                content=raw_text,
                file_path=file_path,
                candidate_name=self._get_author(raw_text, file_path),
                file_ext=FileExtension.TXT,
            )
            self._add_to_disk_cache(document)

            self._print(
                f"[bold bright_green]COMPLETE[/] - [magenta underline]{file_path}[/] completed in [bright_green]{(time.perf_counter() - start_time):0.2f} seconds[/bright_green]"
            )
            return [document]

    def extract_rtf(self, file_path: str) -> list[ExtractedDocument]:
        with open(file_path, "r") as f:
            raw_text = rtf_to_text(f.read())
            persisted = self._get_persisted_document(raw_text)
            if persisted is not None:
                self._print(f"[bold green]FOUND[/] - [magenta underline]{file_path}[/]")
                return [persisted]

            start_time = time.perf_counter()

            self._print(
                f"[bold orange]PROCESSING[/] - [magenta underline]{file_path}[/] ..."
            )
            document = ExtractedDocument(
                content=rtf_to_text(raw_text),
                file_path=file_path,
                candidate_name=self._get_author(raw_text, file_path),
                file_ext=FileExtension.RTF,
            )
            self._add_to_disk_cache(document)

            self._print(
                f"[bold bright_green]COMPLETE[/] - [magenta underline]{file_path}[/] completed in [bright_green]{(time.perf_counter() - start_time):0.2f} seconds[/bright_green]"
            )
            return [document]

    def extract_zip(self, file_path: str) -> list[ExtractedDocument]:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            files = zip_ref.namelist()
            root = file_path[: file_path.rindex("/")]
            zip_ref.extractall(root)

            parsed = [
                x for file in files for x in self.extract(os.path.join(root, file))
            ]

            # Cleanup
            for file in files:
                os.remove(os.path.join(root, file))

            return parsed

    @cache
    def extract(self, file_path: str) -> list[ExtractedDocument]:
        ext = get_file_ext(file_path)
        match ext:
            case FileExtension.PDF:
                return self.extract_pdf(file_path)
            case FileExtension.DOCX:
                return self.extract_docx(file_path)
            case FileExtension.DOC:
                return self.extract_doc(file_path)
            case FileExtension.TXT:
                return self.extract_txt(file_path)
            case FileExtension.RTF:
                return self.extract_rtf(file_path)
            case FileExtension.ZIP:
                return self.extract_zip(file_path)
            case _:
                raise ValueError(
                    f"Unsupported file extension: \"{file_path[file_path.rindex('.') + 1:]}\""
                )
