import os
import csv
import xml.etree.ElementTree as ET
from tqdm import tqdm

MEDQUAD_ROOT = "MedQuAD"
OUTPUT_CSV = "data/medquad.csv"

def clean_text(text: str) -> str:
    """Normalize whitespace and newlines"""
    return " ".join(text.split()).strip()

def parse_medquad_to_csv():
    rows = []

    for root, _, files in os.walk(MEDQUAD_ROOT):
        for file in files:
            if not file.endswith(".xml"):
                continue

            file_path = os.path.join(root, file)

            try:
                tree = ET.parse(file_path)
                doc = tree.getroot()

                qa_pairs = doc.find("QAPairs")
                if qa_pairs is None:
                    continue

                for qa in qa_pairs.findall("QAPair"):
                    q_el = qa.find("Question")
                    a_el = qa.find("Answer")

                    if q_el is None or a_el is None:
                        continue

                    question = clean_text(q_el.text or "")
                    answer = clean_text(" ".join(a_el.itertext()))

                    if not question or not answer:
                        continue

                    rows.append({
                        'prompt': question,
                        'response': answer
                    })

            except Exception as e:
                print(f"❌ Error parsing {file_path}: {e}")

    # ---- Write CSV ----
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['prompt', 'response'],
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} Q&A pairs to {OUTPUT_CSV}")

if __name__ == "__main__":
    parse_medquad_to_csv()
