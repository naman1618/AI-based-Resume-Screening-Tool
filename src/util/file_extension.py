from enum import Enum


class FileExtension(Enum):
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    RTF = "rtf"
    ZIP = "zip"
    UNSUPPORTED = 0xDEADBEEF


def get_file_ext(file_name: str) -> FileExtension:
    file_name = file_name.strip().lower()
    ext = file_name[file_name.rindex(".") + 1 :]
    try:
        return FileExtension.__dict__["_value2member_map_"][ext]
    except:
        return FileExtension.UNSUPPORTED
