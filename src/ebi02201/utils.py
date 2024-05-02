import requests
import concurrent.futures
import re
from bs4 import BeautifulSoup
import spacy
import aiohttp

session = requests.Session()

# Load the Spacy model outside the function to avoid reloading it each time the function is called
nlp = spacy.load("en_core_sci_sm", disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
nlp.add_pipe("sentencizer")

async def fetch_xml(pmcid):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text()

def get_relevant_paragraphs(pmcid, partial_sentences):
    relevant_paragraphs = []
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml-xml')
        p_tags = soup.find_all('p')

        # Compile the regular expression pattern for efficiency
        pattern = re.compile('|'.join(map(re.escape, partial_sentences)))

        for tag in p_tags:
            paragraph_text = tag.get_text()
            if pattern.search(paragraph_text):
                relevant_paragraphs.append(paragraph_text)

        return relevant_paragraphs
    except requests.RequestException:
        return None

def segment_sentences_spacy(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def get_full_text_xml_paragraphs(pmcid, partial_sentences):
    segmented_sentences = []
    relevant_paragraphs = get_relevant_paragraphs(pmcid, partial_sentences)

    if relevant_paragraphs:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(segment_sentences_spacy, paragraph) for paragraph in relevant_paragraphs]
            for future in concurrent.futures.as_completed(futures):
                segmented_sentences.extend(future.result())

    return segmented_sentences

def balance_ner_samples(df):
    min_count = df['ner'].value_counts().min()
    balanced_df = df.groupby('ner').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    return balanced_df

def find_sentence_with_substring(string_list, substring):
    pattern = re.compile(r'(?<=[.!?])\s+')
    for text in string_list:
        sentences = pattern.split(text)
        for sentence in sentences:
            if substring in sentence:
                return sentence
    return None

def process_pmcid(df, pmcid, p_texts):
    sentences_data = {}
    for row in df.itertuples(index=False):
        if row.pmc_id != pmcid:
            continue
        sentence = find_sentence_with_substring(p_texts, row.partial_sentence)
        if sentence:
            token = row.token
            start = 0
            while start != -1:
                start = sentence.find(token, start)
                if start != -1:
                    end = start + len(token)
                    sentences_data.setdefault(sentence, set()).add((start, end, token, row.ner))
                    start += len(token)  # Move past the last found token

    return [[pmcid, sentence, list(ner_tags)] for sentence, ner_tags in sentences_data.items()]


def find_sub_span(token_span, entity_span):
    start, end = max(token_span[0], entity_span[0]), min(token_span[1], entity_span[1])
    return (start, end) if start < end else None

def process_articles(pmcids, annotations_df):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_pmcid = {executor.submit(get_full_text_xml_paragraphs, pmcid,
                                           annotations_df[annotations_df['pmc_id'] == pmcid]['partial_sentence'].tolist()): pmcid 
                           for pmcid in pmcids}

        results = []
        for future in concurrent.futures.as_completed(future_to_pmcid):
            pmcid = future_to_pmcid[future]
            try:
                segmented_sentences = future.result()
                processed_data = process_pmcid(annotations_df, pmcid, segmented_sentences)
                results.extend(processed_data)
            except Exception as exc:
                print(f'{pmcid} generated an exception: {exc}')
        return results