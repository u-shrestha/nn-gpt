
import os
import json
from datetime import datetime
import ab.nn.api as api
from ab.nn.util.Util import read_py_file_as_string
import ab.nn.util.db as db

from pathlib import Path


# Configuration
TRANSFORM_DIR = "ab/gpt/brute/trans/transform_files"

def populate_transform_files():
    from pathlib import Path
    conn, cursor = db.Init.sql_conn()
    
    transform_dir = Path(TRANSFORM_DIR)
    transform_files = [f for f in transform_dir.iterdir() if f.is_file() and f.suffix == ".py"]

    for code_file in transform_files:
        db.Write.code_to_db(cursor, "transform", code_file=code_file) 

    db.Init.close_conn(conn)
    print("Transform files added/updated successfully.")


if __name__ == "__main__":
    populate_transform_files()
