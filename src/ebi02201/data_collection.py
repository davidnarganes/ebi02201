import asyncio
import aiohttp
import pandas as pd
import spacy

# Load Spacy model outside the function
nlp = spacy.load("en_core_sci_sm", disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
nlp.add_pipe("sentencizer")

async def fetch_data_for_type_async(session: aiohttp.ClientSession, annotation_type: str, cursor_mark: str, page_size: int) -> dict:
    """
    Fetch data for a specific annotation type asynchronously.
    """
    base_url = "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsBySectionAndOrType"
    params = {
        'type': annotation_type,
        'filter': 1,
        'format': 'JSON',
        'cursorMark': cursor_mark,
        'pageSize': page_size
    }
    async with session.get(base_url, params=params) as response:
        if response.status == 200:
            return await response.json()
        else:
            response.raise_for_status()

def process_article(article: dict) -> list:
    """
    Process a single article.
    """
    records = []
    pmc_id = article.get('pmcid')
    if not pmc_id:
        return records

    for annotation in article['annotations']:
        exact = annotation.get('prefix', '') + annotation.get('exact', '') + annotation.get('postfix', '')
        token = annotation['tags'][0]['name'] if annotation['tags'] else ''
        ner = annotation['type']
        records.append([pmc_id, exact, token, ner])
    
    return records

async def get_epmc_annotations_async(annotation_types: list, iterations: int, page_size: int) -> pd.DataFrame:
    """
    Asynchronously get EPMC annotations.
    """
    data_list = []
    async with aiohttp.ClientSession() as session:
        for annotation_type in annotation_types:
            cursor_mark = "0.0"
            for _ in range(iterations):
                data = await fetch_data_for_type_async(session, annotation_type, cursor_mark, page_size)
                cursor_mark = data['nextCursorMark']
                for article in data['articles']:
                    data_list.extend(process_article(article))

    return pd.DataFrame(data_list, columns=['pmc_id', 'partial_sentence', 'token', 'ner'])

async def main():
    annotation_types = ['Gene Mutations', 'Cell', 'Cell Line', 'Organ Tissue']
    annotations_df = await get_epmc_annotations_async(annotation_types, iterations=2, page_size=4)
    annotations_df.to_csv('../../data/raw/annotation_df.csv', index=False)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())