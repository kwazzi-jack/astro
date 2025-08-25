import subprocess as sbp
from pathlib import Path

import mdformat


def format_md_file(file_path: Path):
    try:
        with open(file_path) as file:
            content = file.read()

        fixed_content = (
            content.replace(r"\\", "\\").replace("{{", "{").replace("}}", "}")
        )
        formatted_content = mdformat.text(fixed_content)
        print("    > File formatted")
        with open(file_path, "w") as file:
            file.write(formatted_content)
    except Exception as error:
        print(f"    > Error during markdown formatting: {error}")
        exit(1)


def convert_md_to_pdf(file: Path):
    new_file = file.with_suffix(".pdf")

    result = sbp.run(
        [
            "pandoc",
            f"{file}",
            "-o",
            str(new_file),
            "--pdf-engine=xelatex",
            "--variable",
            "mainfont=Latin Modern Roman",
            "--variable",
            "monofont=Latin Modern Mono",
            "--variable",
            "mathfont=Latin Modern Math",
            "--variable",
            "geometry:margin=1cm",
        ],
        stdout=sbp.PIPE,
        stderr=sbp.PIPE,
    )
    return result


def main():
    experiment_name = "chat-system-2"
    outputs_to_convert = [
        "codellama-latest",
        "llama3-latest",
        "llava-13b",
        "mistral-7b",
        "gpt-4o",
        "gpt-4o-mini",
        "deepseek-r1-14b",
        "deepseek-r1-8b",
    ]
    outputs_dir = Path("./outputs/")
    files = list(outputs_dir.glob("*.md", case_sensitive=True))
    print(f"Output directory has {len(files)} markdown files")
    for i, file in enumerate(files):
        print(f"#{i + 1} {file.name}")
        file_path = None
        for name in outputs_to_convert:
            if name in file.name and experiment_name in file.name:
                file_path = file
                print("    > Matched")
                break
        if file_path is None:
            print("    > File not listed - ignoring")
            continue

        try:
            format_md_file(file)
            result = convert_md_to_pdf(file)
            if result.returncode or not file.exists():
                print(f"    > Error during conversion (return code {result})")
                print(f"OUTPUT:\n{result.stdout}\n\nERROR:\n{result.stderr}")
                exit(1)
            else:
                print("    > Conversion completed")

        except Exception as error:
            print(f"    > Error occurred: {error}")
            exit(1)
    print("Completed")


if __name__ == "__main__":
    main()
