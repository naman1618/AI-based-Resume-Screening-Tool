from llama_index.core import Document as LlamaDocument
from llama_index.readers.web import SimpleWebPageReader
import pymupdf4llm
import mammoth
from markdownify import markdownify as html2md
from enum import Enum


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

    def __init__(self):
        self._pdf_reader = pymupdf4llm.LlamaMarkdownReader()

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

    def load_pdf(self, file_path: str) -> list[LlamaDocument]:
        self._validate_extension_for_loader(file_path, FileExtension.PDF)

        return self._pdf_reader.load_data(file_path)

    def load_docx(self, file_path: str) -> list[LlamaDocument]:
        self._validate_extension_for_loader(file_path, FileExtension.DOCX)
        with open(file_path, "rb") as docx_file:
            md = html2md(mammoth.convert_to_html(docx_file).value)


loader = FileLoader()
docx = loader.load_docx("resumes/Zoha_Resume_R.docx")

print(docx)

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
