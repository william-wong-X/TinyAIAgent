import os

def get_dir(relative_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    full_path = os.path.abspath(os.path.join(project_root, relative_path))
    if not os.path.isdir(full_path):
        raise FileNotFoundError(f"Path is not exist: {full_path}")
    return full_path
