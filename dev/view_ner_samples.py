import pandas as pd
import spacy

def main():
    print("Loading SpaCy zh_core_web_sm...")
    nlp = spacy.load("zh_core_web_sm")
    
    print("Loading Dataset...")
    df = pd.read_parquet("data/Basic_ZH/wiki_filtered.parquet").head(20)
    texts = df['completion'].tolist()
    
    target_labels = {'PERSON', 'LOC', 'ORG', 'GPE', 'FAC'}
    
    print("\n--- NER Extracted Samples ---\n")
    for idx, text in enumerate(texts):
        # truncate for display
        display_text = text[:300].replace('\n', ' ')
        doc = nlp(display_text)
        
        entities = []
        for ent in doc.ents:
            if ent.label_ in target_labels:
                entities.append(f"[{ent.text} ({ent.label_})]")
                
        if entities:
            print(f"Row {idx}:")
            print(f"Text snippet: {display_text}")
            print(f"Extracted Entities: {', '.join(entities)}")
            print("-" * 60)

if __name__ == '__main__':
    main()
