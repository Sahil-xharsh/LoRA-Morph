#Force IPv4 for the upload script
import socket

_original_getaddrinfo = socket.getaddrinfo

def ipv4_getaddrinfo(host, *args, **kwargs):
    results = _original_getaddrinfo(host, *args, **kwargs)
    ipv4_results = [
        result for result in results
        if result[0] == socket.AF_INET
    ]
    return ipv4_results or results

socket.getaddrinfo = ipv4_getaddrinfo
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def list_files(folder: Path):
    """Return files and their total byte size in deterministic order."""
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    return files, sum(path.stat().st_size for path in files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    files, total_size = list_files(args.folder)
    for path in files:
        print(f"{path.relative_to(args.folder)} ({path.stat().st_size} bytes)")
    print(f"total_size={total_size} bytes")
    if args.dry_run:
        return
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id, folder_path=str(args.folder), repo_type="model"
    )


if __name__ == "__main__":
    main()
