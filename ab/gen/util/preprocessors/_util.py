def read_file(path: str) -> str:
    file = open(path, "r")
    out = file.read()
    file.close()
    return out
