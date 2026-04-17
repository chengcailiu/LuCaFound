import argparse
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


LABEL_KEYS = [
    "Pulmonary Micronodules",
    "Pulmonary glass-ground nodules",
    "Old Lesions in Lungs",
    "Aortic and Coronary Artery Calcification",
    "Fatty Liver",
    "Pulmonary Emphysema",
    "Pulmonary Bullae",
    "Liver Cysts",
    "Localized Pleural Thickening",
    "Low-Density Shadow in Thyroid",
    "Renal Cysts or Stones",
    "Incomplete Thymic Involution",
    "Thickening of Adrenal Gland",
    "Gallstones",
    "Calcification of Mediastinal Lymph Nodes",
    "Pulmonary Infection or Inflammation",
    "Liver Calcification",
    "Thyroid Nodules",
    "Pericardial Effusion",
    "Bronchiectasis",
]


PROMPT_TEMPLATE = """You are helping convert raw chest CT reports into a fixed 20-class finding summary.

Task:
Read the input report and determine whether each of the following findings is positive or negative:
{label_list}

Output rules:
1. Output exactly one line for each finding.
2. Keep the same order as the provided list.
3. Use the exact format: <finding name>-positive or <finding name>-negative
4. Do not output explanations, numbering, or extra text.

Input report:
{report_text}
"""


def build_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Failed to decode {path}")


def infer_output_name(input_name: str) -> str:
    if input_name.endswith("_text_ori.txt"):
        return input_name.replace("_text_ori.txt", "_text_anomal.txt")
    if input_name.endswith("_ori.txt"):
        return input_name.replace("_ori.txt", "_text_anomal.txt")
    if input_name.endswith(".txt"):
        return input_name.replace(".txt", "_text_anomal.txt")
    return f"{input_name}_text_anomal.txt"


def iter_input_files(input_dir: Path):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".txt":
            yield path


def generate_label_text(client: OpenAI, model: str, report_text: str, temperature: float) -> str:
    label_list = "\n".join(f"- {item}" for item in LABEL_KEYS)
    prompt = PROMPT_TEMPLATE.format(label_list=label_list, report_text=report_text)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful medical text structuring assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def normalize_label_name(label_name: str) -> str:
    label_name = label_name.strip().lower()
    label_name = label_name.replace("_", " ")
    label_name = re.sub(r"\s+", " ", label_name)
    return label_name


def clean_output_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*[-*•]+\s*", "", line)
    line = re.sub(r"^\s*\d+[\.\)、:：-]?\s*", "", line)
    return line.strip()


def validate_and_format_label_text(raw_output: str) -> str:
    canonical_label_map = {normalize_label_name(label): label for label in LABEL_KEYS}
    parsed = {}

    for raw_line in raw_output.splitlines():
        line = clean_output_line(raw_line)
        if not line:
            continue

        matched = re.match(r"^(?P<label>.+?)\s*[-:：]\s*(?P<status>positive|negative)\s*$", line, flags=re.IGNORECASE)
        if not matched:
            raise ValueError(f"Invalid output line format: {raw_line!r}")

        label_name = normalize_label_name(matched.group("label"))
        status = matched.group("status").lower()

        if label_name not in canonical_label_map:
            raise ValueError(f"Unknown label name in output: {matched.group('label')!r}")

        canonical_label = canonical_label_map[label_name]
        if canonical_label in parsed:
            raise ValueError(f"Duplicate label found in output: {canonical_label}")

        parsed[canonical_label] = status

    missing_labels = [label for label in LABEL_KEYS if label not in parsed]
    if missing_labels:
        raise ValueError(f"Missing labels in output: {missing_labels}")

    extra_count = len(parsed) - len(LABEL_KEYS)
    if extra_count != 0:
        raise ValueError(f"Unexpected number of parsed labels: {len(parsed)}")

    return "\n".join(f"{label}-{parsed[label]}" for label in LABEL_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 20-class finding text from original CT reports using an LLM API."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing original report text files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated 20-class finding text files.")
    parser.add_argument("--model", default="deepseek-chat", help="Chat model name.")
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"), help="API base URL.")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""), help="API key. Prefer setting OPENAI_API_KEY.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files that already have generated outputs.")
    parser.add_argument("--sleep_seconds", type=float, default=0.0, help="Optional delay between API calls.")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api_key.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(api_key=args.api_key, base_url=args.base_url)
    input_files = list(iter_input_files(input_dir))

    if not input_files:
        raise ValueError(f"No .txt files found in {input_dir}")

    for report_path in tqdm(input_files, desc="Generating 20-class texts"):
        output_name = infer_output_name(report_path.name)
        output_path = output_dir / output_name

        if args.skip_existing and output_path.exists():
            continue

        report_text = read_text_file(report_path)
        label_text = generate_label_text(
            client=client,
            model=args.model,
            report_text=report_text,
            temperature=args.temperature,
        )
        validated_label_text = validate_and_format_label_text(label_text)
        output_path.write_text(validated_label_text + "\n", encoding="utf-8")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
