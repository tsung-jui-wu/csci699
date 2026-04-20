import os
import re
import sys
import fitz  # PyMuPDF


# Patterns for abstract end-boundary (section headers that follow the abstract)
_INTRO_PATTERN = re.compile(
    r"^(1[\s.\-]*\s*(introduction|motivation|background|overview|related\s+work)"
    r"|I[\s.\-]+introduction"
    r"|introduction)$",
    re.I,
)


def find_abstract_body_rect(page):
    """
    Find the bounding rect of the abstract body on this page (excluding the header).
    Returns (abstract_body_rect, intro_y_or_None, orig_fontsize).
    Raises ValueError if the Abstract header is not found or no body blocks exist.
    """
    blocks = page.get_text("blocks")
    # Each block: (x0, y0, x1, y1, text, block_no, block_type)

    abstract_idx = None
    intro_y = None

    for i, b in enumerate(blocks):
        text = b[4].strip()
        # Match header variants: "Abstract", "ABSTRACT", "A B S T R A C T"
        if abstract_idx is None and re.match(
            r"^a[\s]*b[\s]*s[\s]*t[\s]*r[\s]*a[\s]*c[\s]*t\b", text, re.I
        ):
            abstract_idx = i
        elif abstract_idx is not None and _INTRO_PATTERN.match(text):
            intro_y = b[1]  # y0 of the first section header after abstract
            break

    if abstract_idx is None:
        raise ValueError(
            "Could not find an 'Abstract' header in the PDF. "
            "The abstract may be in a non-standard format."
        )

    # Collect body blocks after the header, stopping at the intro boundary
    body_blocks = blocks[abstract_idx + 1 :]
    if intro_y is not None:
        body_blocks = [b for b in body_blocks if b[1] < intro_y]

    # Strip keywords lines
    body_blocks = [
        b
        for b in body_blocks
        if not re.match(r"^(keywords|key\s*words)\b", b[4].strip(), re.I)
    ]

    if not body_blocks:
        raise ValueError("Found 'Abstract' header but no body text blocks beneath it.")

    # Build union rect
    rect = fitz.Rect(body_blocks[0][:4])
    for b in body_blocks[1:]:
        rect |= fitz.Rect(b[:4])

    # Detect original font size from the first span inside the body rect
    orig_fontsize = 10.0  # fallback
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        br = fitz.Rect(block["bbox"])
        if rect.intersects(br) and br.y0 > fitz.Rect(blocks[abstract_idx][:4]).y1:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("size"):
                        orig_fontsize = span["size"]
                        break
                else:
                    continue
                break
            else:
                continue
            break

    return rect, intro_y, orig_fontsize


def replace_abstract(input_path: str, output_path: str, new_text: str) -> None:
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Output path must differ from input path.")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(out_dir):
        raise ValueError(f"Output directory does not exist: {out_dir}")

    doc = fitz.open(input_path)
    try:
        if doc.page_count == 0:
            raise ValueError("PDF contains no pages.")

        page = doc[0]
        abstract_rect, intro_y, fontsize = find_abstract_body_rect(page)

        if intro_y is None:
            print(
                "Warning: no section header found after Abstract — redacting all text "
                "below the Abstract header on page 1. Review the output carefully.",
                file=sys.stderr,
            )

        # Warn if abstract may continue onto page 2
        page_height = page.rect.height
        if intro_y is None and abstract_rect.y1 >= page_height - 20:
            print(
                "Warning: abstract may continue onto page 2; only page 1 is modified.",
                file=sys.stderr,
            )

        page.add_redact_annot(abstract_rect, fill=(1, 1, 1))
        page.apply_redactions()

        overflow = page.insert_textbox(
            abstract_rect,
            new_text,
            fontname="tiro",  # Times Roman — closest base-14 match to Computer Modern
            fontsize=fontsize,
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_JUSTIFY,
        )

        if overflow < 0:
            print(
                f"Warning: replacement text overflowed the abstract box by "
                f"{abs(overflow):.1f}pt. Consider shortening the text.",
                file=sys.stderr,
            )

        doc.save(output_path, garbage=4, deflate=True)

    finally:
        doc.close()

    print(f"Saved revised PDF to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Replace the abstract in a research paper PDF.")
    parser.add_argument(
        "-i", "--input",
        default="input.pdf",
        help="Input PDF path (default: input.pdf)",
    )
    parser.add_argument(
        "-o", "--output",
        default="output.pdf",
        help="Output PDF path (default: output.pdf)",
    )
    parser.add_argument(
        "-t", "--text",
        required=True,
        help="Path to a .txt file containing the replacement abstract text",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.text):
        print(f"Error: text file not found: {args.text}", file=sys.stderr)
        sys.exit(1)

    with open(args.text, encoding="utf-8") as f:
        new_text = f.read().strip()

    if not new_text:
        print("Error: text file is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        replace_abstract(args.input, args.output, new_text)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
