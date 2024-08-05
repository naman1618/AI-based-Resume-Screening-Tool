import time
from llama_index.core import Document as LlamaDocument
import mammoth
from markdownify import markdownify as html2md
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from md2propositions import Markdown2Propositions
from pypdf import PdfReader
from text2md import Text2Markdown


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
        self._verbose = verbose
        self._text2md = Text2Markdown(verbose=verbose)
        self._md2propositions = Markdown2Propositions(verbose=verbose)

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

    def load(self, file_path: str) -> tuple[list[str], str, str]:
        """
        Loads a document, calling the appropriate load method based on its extension

        Returns:
            A tuple containing a list of propositions and the markdown hash id
        """
        match self.get_extension(file_path):
            case FileExtension.PDF:
                return self.load_pdf(file_path)
            case FileExtension.DOCX:
                return self.load_docx(file_path)
            case _:
                return []

    def _print(self, string):
        if self._verbose:
            print(string)

    def load_pdf(self, file_path: str) -> tuple[list[str], str, str]:
        self._validate_extension_for_loader(file_path, FileExtension.PDF)

        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            raw_text = "\n".join([x.extract_text() for x in reader.pages])
            return self._rawtext2propositions(raw_text=raw_text, file_path=file_path)

    def load_docx(self, file_path: str) -> tuple[list[str], str, str]:
        self._validate_extension_for_loader(file_path, FileExtension.DOCX)
        with open(file_path, "rb") as docx_file:
            raw_text = mammoth.extract_raw_text(docx_file).value
            return self._rawtext2propositions(raw_text=raw_text, file_path=file_path)

    def _rawtext2propositions(
        self, raw_text: str, file_path: str
    ) -> tuple[list[str], str, str]:
        start = time.perf_counter()
        self._print(f"Generating markdown version of {file_path}...")
        output = self._text2md.to_markdown(raw_text)
        self._print(f"Done. Took {time.perf_counter() - start} seconds")
        self._print(f"Getting the propositions...")
        start = time.perf_counter()
        propositions, name = self._md2propositions.get_propositions(output)
        self._print(f"Done. Took {time.perf_counter() - start} seconds")
        return (propositions, output[1], name)
