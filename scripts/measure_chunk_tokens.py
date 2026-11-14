import argparse
from pathlib import Path
from transformers import AutoTokenizer
from doubtdesk.ingestion.extractor import extract_document
from doubtdesk.ingestion.chunking import chunk_document
from doubtdesk.ingestion.schemas import PageRecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdfs', nargs='+', type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    total = truncated = maximum = 0
    for path in args.pdfs:
        pages = [PageRecord(doc_id='sample', image_path='', **page) for page in extract_document(path)]
        counts = [len(tokenizer(chunk.text, truncation=False)['input_ids']) for chunk in chunk_document(pages)]
        over = sum(count > 256 for count in counts)
        total += len(counts)
        truncated += over
        maximum = max([maximum, *counts])
        print(f'{path.name}: {over}/{len(counts)} chunks exceed 256 tokens')
    print(f'Total: {truncated}/{total} ({100*truncated/max(total,1):.2f}%), max={maximum}')


if __name__ == '__main__':
    main()
