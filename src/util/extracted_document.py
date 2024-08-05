from pydantic import BaseModel
from util.file_extension import FileExtension


class ExtractedDocument(BaseModel):
    content: str
    file_path: str
    candidate_name: str
    file_ext: FileExtension

    class Config:
        json_encoders = {FileExtension: lambda x: x.value}

    def __hash__(self):
        return hash(self.content)
