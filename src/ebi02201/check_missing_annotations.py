import pandas as pd
import re
import collections

# Read the DataFrame and evaluate the 'ner' column
df = pd.read_csv('../../data/raw/biggest_test.csv')
df['ner'] = df['ner'].apply(eval)

# Define the sentence you want to check
# From PMC3500179
sentence_to_check = '12 strains containing different drug-resistant mutation'
sentence_to_check = 'In 49 families, 50 new mutations'

# Filter the DataFrame to get the sentence and its corresponding annotations
mask_check = df['sentence'].str.startswith(sentence_to_check)
sentence, annotations = df.loc[mask_check, ['sentence', 'ner']].values[0]

# Extract the entities from the annotations
texts = [x[-2] for x in annotations]

# Count the occurrences of unique entities
entity_counts = collections.Counter(texts)

# Iterate through unique entities and extract their start and end positions
unique_entities = set(texts)
matching_spans = set()

for entity in unique_entities:
    # Use regex to find all matches and their start and end positions
    matches = [(match.start(), match.end()) for match in re.finditer(re.escape(entity), sentence)]
    
    print(f"Entity: {entity}")
    print(matches)
    
    for match in matches:
        matching_spans.add((match[0], match[1], entity))

# Create a set from the annotations for comparison
annotations_formatted = {tuple(x[:-1]) for x in annotations}

# Calculate missing spans
missing_spans = matching_spans - annotations_formatted

# Calculate intersection
intersection = matching_spans.intersection(annotations_formatted)

# Calculate union
union = matching_spans.union(annotations_formatted)

# Calculate disunion (symmetric difference)
disunion = matching_spans.symmetric_difference(annotations_formatted)

# Calculate unique elements in each set
unique_matching_spans = matching_spans - annotations_formatted
unique_annotations_formatted = annotations_formatted - matching_spans

# Display the results
print("\nMissing Spans:")
print(missing_spans)

print("\nIntersection:")
print(intersection)

print("\nUnion:")
print(union)

print("\nDisunion (Symmetric Difference):")
print(disunion)

print("\nUnique in Matching Spans:")
print(unique_matching_spans)

print("\nUnique in Annotations Formatted:")
print(unique_annotations_formatted)
