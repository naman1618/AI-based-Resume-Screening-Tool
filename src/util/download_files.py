from pydantic import BaseModel
import os
from constants import DOWNLOAD_DIR_MAPPING, BUCKET_NAME
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor


class DownloadOperationStatus(BaseModel):
    success: bool
    error: str | None
    file_path: str


def download(blob):
    try:
        # Use .index instead of .rindex to preserve nested directories (e.g. transcripts/NestedDir/hired.pdf)
        file_name = blob.name[blob.name.rindex("/") + 1 :]
        src_parent_dir = blob.name[: blob.name.index("/")]
    except:
        # Means that this file is in the root directory
        file_name = blob.name
        src_parent_dir = "."
    dest_parent_dir = DOWNLOAD_DIR_MAPPING[src_parent_dir]
    download_path = f"{dest_parent_dir}/{file_name}"
    if os.path.exists(download_path):
        print(f"{blob.name} already exists in {dest_parent_dir}. Skipping...")
        return DownloadOperationStatus(success=True, file_path=blob.name, error=None)

    try:
        parent_dir_path = download_path[: download_path.rindex("/")]
        os.makedirs(parent_dir_path, exist_ok=True)
    except:
        pass

    if not os.path.isdir(blob.name):
        try:
            print(f"Downloading {file_name} to {download_path}")
            blob.download_to_filename(download_path)
            print(f"\t- {download_path} complete")
        except Exception as e:
            return DownloadOperationStatus(
                success=False, file_path=blob.name, error=str(e)
            )
    return DownloadOperationStatus(success=True, file_path=blob.name, error=None)


def main():
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blobs = client.list_blobs(BUCKET_NAME)

    # Downloads all resume files if they do not exist. If they do exist, they are skipped

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = [x for x in executor.map(download, blobs)]
        failed: list[tuple[str, str]] = [
            (result.file_path, result.error) for result in results if not result.success
        ]

        if len(failed) > 0:
            print(
                f"{len(failed)} file(s) failed to download\n\nThe full list of files that failed to download: {[x[0] for x in failed]}"
            )
            for failure in failed:
                print(f"{failure[0]} failed to download!")
                print(failure[1])
        print(f"{len(results) - len(failed)}/{len(results)} files loaded")
