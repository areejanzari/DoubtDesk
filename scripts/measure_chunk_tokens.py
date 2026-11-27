import argparse
import json
from pathlib import Path
import pymupdf
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from doubtdesk.ingestion.chunking import chunk_document
from doubtdesk.ingestion.embeddings import EMBEDDING_REPO
from doubtdesk.ingestion.extractor import extract_page_content
from doubtdesk.ingestion.layout import clean_document
from doubtdesk.ingestion.schemas import PageRecord


def document_pages(path: Path) -> list[PageRecord]:
    with pymupdf.open(path) as pdf:
        contents = [extract_page_content(page) for page in pdf]
    texts = clean_document([content["lines"] for content in contents])
    return [
        PageRecord(doc_id='sample', page_number=number, text=text, classification=content["classification"],
                   image_coverage=content["image_coverage"])
        for number, (content, text) in enumerate(zip(contents, texts), 1)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdfs', nargs='+', type=Path)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(hf_hub_download(EMBEDDING_REPO, 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    with open(hf_hub_download(EMBEDDING_REPO, 'sentence_bert_config.json'), encoding='utf-8') as handle:
        limit = json.load(handle)['max_seq_length']
    total = truncated = maximum = 0
    for path in args.pdfs:
        counts = [len(tokenizer.encode(chunk.text).ids) for chunk in chunk_document(document_pages(path), tokenizer)]
        over = sum(count > limit for count in counts)
        total += len(counts)
        truncated += over
        maximum = max([maximum, *counts])
        print(f'{path.name}: {over}/{len(counts)} chunks exceed {limit} tokens')
    print(f'Total: {truncated}/{total} ({100*truncated/max(total,1):.2f}%), max={maximum}')


if __name__ == '__main__':
    main()
