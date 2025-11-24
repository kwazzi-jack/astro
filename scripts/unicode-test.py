def generate_cases(*cases: str) -> str:
    count = len(cases)
    if count == 0:
        return ""

    top_i = 0
    bot_i = count - 1
    diff = bot_i - top_i
    print(diff)
    # One line
    if diff == 0:
        return f"{{{cases[0]}"

    result = []
    # Two line
    if diff == 1:
        result.append(f"⎛{cases[0]}")
        result.append("⎨")
        result.append(f"⎝{cases[1]}")
    elif diff == 2:
        result.append(f"⎛{cases[0]}")
        result.append(f"⎨{cases[1]}")
        result.append(f"⎝{cases[2]}")
    elif diff % 2 == 1:
        mid = (diff - 1) // 2
        result.append(f"⎛{cases[0]}")
        for case in cases[1:mid]:
            result.append(f"⎜{case}")
        result.append(f"⎨{cases[mid]}")
        for case in cases[mid + 1 : -1]:
            result.append(f"⎜{case}")
        result.append(f"⎝{cases[-1]}")
    else:
        mid = (diff - 1) // 2 + 1
        result.append(f"⎛{cases[0]}")
        for case in cases[1:mid]:
            result.append(f"⎜{case}")
        result.append(f"⎨{cases[mid]}")
        for case in cases[mid + 1 : -1]:
            result.append(f"⎜{case}")
        result.append(f"⎝{cases[-1]}")
    return "\n".join(result)


def fill_text(text: str, pad_bot: int, pad_top: int) -> str:
    lines = text.splitlines()
    max_length = max(len(line.strip()) for line in lines)
    pad = " " * max_length
    padded_lines = (
        [pad] * pad_top
        + [line + " " * (max_length - len(line)) for line in lines]
        + [pad] * pad_bot
    )
    return "\n".join(padded_lines)


def join_horizontally(text1: str, text2: str) -> str:
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    if len(lines1) < len(lines2):
        diff = len(lines2) - len(lines1)
        if diff % 2 == 0:
            pad_top = pad_bot = diff // 2
        else:
            pad_top = diff // 2
            pad_bot = diff // 2 + 1

        text1 = fill_text(text1, pad_bot, pad_top)
        lines1 = text1.splitlines()

    elif len(lines2) < len(lines1):
        diff = len(lines2) - len(lines1)
        if diff % 2 == 0:
            pad_top = pad_bot = diff // 2
        else:
            pad_top = diff // 2
            pad_bot = diff // 2 + 1

        text2 = fill_text(text2, pad_bot, pad_top)
        lines2 = text2.splitlines()

    return "\n".join(f"{line1}{line2}" for line1, line2 in zip(lines1, lines2))


for n in range(1, 11):
    prefix = "f^2(x)="
    cases = generate_cases(*[f"{i**2}, x={i}" for i in range(n)])
    func = join_horizontally(prefix, cases)
    print(f"Testing: {n=}")
    print(func)
    print()
