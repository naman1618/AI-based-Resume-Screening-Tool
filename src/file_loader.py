import time
from llama_index.core import Document as LlamaDocument
import pymupdf4llm
import mammoth
from markdownify import markdownify as html2md
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from parsers import (
    md_json_parser,
    information_integrity_agent_output_parser,
    extract_person_metadata,
)
from prompts import PROPOSITION_GENERATOR_PROMPT, INFORMATION_INTEGRITY_PROMPT
from operator import itemgetter
from pypdf import PdfReader
import os


class UnsupportedFileException(Exception):
    def _extract_extension(self, file_path: str):
        return file_path[file_path.rfind(".") + 1 :]

    def __init__(self, file_path: str, extension: str | None = None) -> None:
        supported_extensions = ", ".join(
            [
                ext
                for ext in FileExtension.__dict__["_value2member_map_"].keys()
                if type(ext) == str
            ]
        )
        super().__init__(
            f'The file "{file_path}" is not supported by the loader. The extension "{self._extract_extension(file_path)}" is not among the supported extensions: {supported_extensions}'
            if extension == None
            else f'The file "{file_path}" is not supported by the loader. The extension "{self._extract_extension(file_path)}" must be "{extension}"'
        )


class FileExtension(Enum):
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    TXT = "txt"
    UNSUPPORTED = 0xDEADBEEF


class FileLoader:

    def __init__(self, verbose=False):
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        information_integrity_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INFORMATION_INTEGRITY_PROMPT),
                ("human", "Document 1:\n{markdown}\n\nDocument 2:{raw_text}"),
            ]
        )

        proposition_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    PROPOSITION_GENERATOR_PROMPT,
                ),
                ("human", "Decompose the following:\n\n{input}"),
            ]
        )
        self._verbose = verbose
        self._json_parser = JsonOutputParser()
        self._agentic_splitter = (
            information_integrity_prompt
            | llm
            | information_integrity_agent_output_parser
            | proposition_prompt
            | llm
            | md_json_parser
        )

    def get_extension(self, file_path: str) -> FileExtension:
        """
        Gets the file extension of a given file path or name with extension
        """
        try:
            return FileExtension.__dict__["_value2member_map_"][
                file_path[file_path.rfind(".") + 1 :]
            ]
        except:
            return FileExtension.UNSUPPORTED

    def _validate_extension_for_loader(self, file_path: str, extension: FileExtension):
        if self.get_extension(file_path) == FileExtension.UNSUPPORTED:
            raise UnsupportedFileException(file_path, extension.value)

    def load(self, file_path: str) -> list[LlamaDocument]:
        """
        Loads a document, calling the appropriate load method based on its extension
        """
        match self.get_extension(file_path):
            case FileExtension.PDF.value:
                return self.load_pdf(file_path)
            case FileExtension.DOCX.value:
                return self.load_docx(file_path)
            case _:
                return []

    def _print(self, string):
        if self._verbose:
            print(string)

    def load_pdf(self, file_path: str) -> list[LlamaDocument]:
        self._validate_extension_for_loader(file_path, FileExtension.PDF)

        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            raw_text = "\n".join([x.extract_text() for x in reader.pages])
            md = pymupdf4llm.to_markdown(file_path, page_chunks=True)[0]
            start = time.perf_counter()
            self._print(f"Refining {file_path}...")
            output = self._agentic_splitter.invoke(
                {"markdown": md["text"], "raw_text": raw_text}
            )
            self._print(
                f"Refining {file_path} took {time.perf_counter() - start} seconds. Now extracting metadata based on the contents..."
            )
            start = time.perf_counter()
            metadata = extract_person_metadata(output)
            self._print(
                f"Extracting metadata from {file_path} took {time.perf_counter() - start} seconds. Action completed."
            )

            return [
                LlamaDocument(
                    text=chunk,
                    metadata={"file_path": os.path.abspath(file_path), **metadata},
                )
                for chunk in output
            ]

    def load_docx(self, file_path: str) -> list[LlamaDocument]:
        self._validate_extension_for_loader(file_path, FileExtension.DOCX)
        with open(file_path, "rb") as docx_file:
            raw_text = mammoth.extract_raw_text(docx_file).value
            md = html2md(mammoth.convert_to_html(docx_file).value)
            start = time.perf_counter()
            self._print(f"Refining {file_path}...")
            output = self._agentic_splitter.invoke(
                {"markdown": md, "raw_text": raw_text}
            )
            self._print(
                f"Refining {file_path} took {time.perf_counter() - start} seconds. Now extracting metadata based on the contents..."
            )
            start = time.perf_counter()
            metadata = extract_person_metadata(output)
            self._print(
                f"Extracting metadata from {file_path} took {time.perf_counter() - start} seconds. Action completed."
            )
            return [
                LlamaDocument(
                    text=chunk,
                    metadata={"file_path": os.path.abspath(file_path), **metadata},
                )
                for chunk in output
            ]


# print(docx)

# def load()


# def load_document(file_path: str) -> LlamaDocument:
#     parsed_file_path = file_path.lower()
#     if parsed_file_path.endswith(".pdf"):
#         return load_pdf(file_path)
#     elif parsed_file_path.endswith(".docx"):
#         return load_docx(file_path)
#     elif parsed_file_path.endswith(".doc"):
#         return load_doc(file_path)
#     elif parsed_file_path.endswith(".txt"):
#         return load_txt(file_path)
#     else:
#         print(f"Unsupported file type: {file_path}")
#         return ""


# def load_pdf(file_path: str) -> list[LlamaDocument]:
#     try:
#         return pdf_reader.load_data(file_path=file_path)
#     except Exception as e:
#         print(f"Error processing PDF file {file_path}: {e}")
#         return []


# def load_docx(file_path):
#     try:
#         doc = DocxDocument(file_path)
#         return "\n".join([para.text for para in doc.paragraphs])
#     except Exception as e:
#         print(f"Error processing .docx file {file_path}: {e}")
#         return ""


# def load_doc(file_path):
#     try:
#         with open(file_path, "rb") as doc_file:
#             result = mammoth.extract_raw_text(doc_file)
#         return result.value
#     except Exception as e:
#         print(f"Error processing .doc file {file_path}: {e}")
#         return ""


# def load_txt(file_path):
#     try:
#         with open(file_path, "r", encoding="utf-8") as file:
#             return file.read()
#     except Exception as e:
#         print(f"Error processing .txt file {file_path}: {e}")
#         return ""
