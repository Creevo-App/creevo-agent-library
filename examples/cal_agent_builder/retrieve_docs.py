import base64
import threading
import time
import requests
import os
from dotenv import load_dotenv
load_dotenv()

GITHUB_TOKEN = os.getenv("GH_TOKEN")
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")
FILE_NAMES = {"README.md", "examples/mcp_context7_agent.ipynb",}  # adjust as needed
REFRESH_INTERVAL = 60 * 20  # 20 minutes

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

# _context_cache = ""
# _cache_lock = threading.Lock()


def fetch_repo_context() -> str:
    # Get full file tree in one API call
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/git/trees/HEAD?recursive=1"
    tree = requests.get(url, headers=HEADERS).json().get("tree", [])

    files = [
        item for item in tree
        if item["type"] == "blob"
        and item["path"] in FILE_NAMES
    ]

    parts = []
    for file in files:
        blob = requests.get(file["url"], headers=HEADERS).json()
        content = base64.b64decode(blob["content"]).decode("utf-8", errors="replace")
        parts.append(f"# FILE: {file['path']}\n{content}")

    return "\n\n---\n\n".join(parts)


# def refresh_cache():
#     global _context_cache
#     while True:
#         try:
#             new_context = fetch_repo_context()
#             with _cache_lock:
#                 _context_cache = new_context
#             print("Context cache refreshed.")
#         except Exception as e:
#             print(f"Failed to refresh cache: {e}")
#         time.sleep(REFRESH_INTERVAL)


# def get_context() -> str:
#     with _cache_lock:
#         return _context_cache


# def start_background_refresh():
#     # Do an initial blocking fetch so cache is warm before serving requests
#     global _context_cache
#     _context_cache = fetch_repo_context()

#     thread = threading.Thread(target=refresh_cache, daemon=True)
#     thread.start()