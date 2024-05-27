def dedent(text: str):
    # Split the text into lines
    lines = text.split("\n")

    # Find the minimum indentation (ignoring empty lines)
    min_indent = None
    for line in lines:
        stripped_line = line.lstrip()
        if stripped_line:
            indent = len(line) - len(stripped_line)
            if min_indent is None:
                min_indent = indent
            else:
                min_indent = min(min_indent, indent)

    if min_indent is None:
        # No non-empty lines
        return text

    # Remove the minimum indentation from each line
    dedented_lines = [line[min_indent:] if line.lstrip() else line for line in lines]

    # Join the lines back together
    return "\n".join(dedented_lines)


dd = dedent
