import zlib
from pathlib import Path

from astro.logging.base import get_logger

logger = get_logger("astro.hashing")


def checksum(
    path: str | Path,
    chunk_size: int = -1,
) -> int:
    if isinstance(path, str):
        # Convert str to Path
        path = Path(path)

    if not path.exists():
        # File does not exist
        msg = f"File does not exist: `{path}`"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Run checksum algorithm
    total = 0
    logger.debug(f"Running checksum on '{path.name}' with `{chunk_size=}`")
    with open(path, "rb") as file:
        # Run with chunks if > 0
        while chunk := file.read(chunk_size):
            total = zlib.crc32(chunk, total)

    # Return checksum
    logger.debug(f"Checksum complete: {total}")
    return total


def files_differ(file1: str | Path, file2: str | Path, chunk_size: int = -1) -> bool:
    files_equal = checksum(file1, chunk_size) == checksum(file2, chunk_size)
    # FIXME logger.debug(f"File '{file1.name}' == File '{file2.name}'? {files_equal}")
    return not files_equal


if __name__ == "__main__":
    content1 = """
    # Sample config content
    [server]
    host = localhost
    port = 8080
    debug = true

    [database]
    url = postgres://user:password@localhost/dbname
    timeout = 30

    [logging]
    level = INFO
    file = /var/log/app.log
    """

    content2 = """
    # ML Pipeline Configuration
    [training]
    epochs = 100
    batch_size = 32
    learning_rate = 0.001

    [model]
    architecture = transformer
    hidden_layers = 4
    dropout = 0.2

    [data]
    train_path = /data/training/
    validation_split = 0.15
    augmentation = true
    """

    content3 = (
        content2
        + """
    # Advanced Settings
    [optimization]
    use_mixed_precision = true
    gradient_accumulation = 4
    weight_decay = 0.01

    [hardware]
    gpu_id = 0
    num_workers = 8
    pin_memory = true
    """
    )

    files = [
        (Path("file1.txt"), content1),
        (Path("file2.txt"), content2),
        (Path("file3.txt"), content3),
        (Path("file4.txt"), content2),
    ]

    total = len(files) ** 2

    print("TEXT TEST")
    passed = 0
    for file, content in files:
        with open(file, "w") as f:
            f.write(content)

    for i, (fileA, contentA) in enumerate(files):
        for j, (fileB, contentB) in enumerate(files):
            file_res = files_differ(fileA, fileB)
            cont_res = contentA.strip() == contentB.strip()
            total_res = file_res == cont_res
            if total_res:
                passed += 1
            print(f"({i=}, {j=}) FILE={file_res} CONTENT={cont_res} WORKED={total_res}")
    print(f"Result: {passed}/{total}\n")

    print("BYTE TEST")
    passed = 0
    for file, content in files:
        with open(file, "wb") as f:
            f.write(bytes(content, encoding="utf-8"))

    for i, (fileA, contentA) in enumerate(files):
        for j, (fileB, contentB) in enumerate(files):
            file_res = files_differ(fileA, fileB)
            cont_res = contentA.strip() == contentB.strip()
            total_res = file_res == cont_res
            if total_res:
                passed += 1
            print(f"({i=}, {j=}) FILE={file_res} CONTENT={cont_res} WORKED={total_res}")
    print(f"Result: {passed}/{total}\n")
