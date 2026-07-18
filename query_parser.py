import logging
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QueryParser")

class ClinicalQueryParser:
    def __init__(self):
        logger.info("Loading spaCy NLP model...")
        self.nlp = spacy.load("en_core_web_sm")

    def parse(self, user_query: str) -> dict:
        """
        Parses raw natural language text to extract intent, entities, 
        and structural metadata parameters for targeted search.
        """
        doc = self.nlp(user_query)
        
        # 1. Extract Named Entities (e.g., Dates, Organizations, Numbers, Locations)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "meaning": spacy.explain(ent.label_)
            })

        # 2. Extract domain keywords (Nouns, Adjectives, Proper Nouns) 
        # while stripping out conversational filler (e.g., "show", "me", "find")
        keywords = [
            token.lemma_.lower() 
            for token in doc 
            if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop
        ]

        # 3. Detect Demographic or Categorical hints (Simple rule-matching logic)
        demographics = []
        lower_query = user_query.lower()
        if any(w in lower_query for w in ["adult", "elderly", "aged", "over 50", "senior"]):
            demographics.append("Adult/Mature")
        if any(w in lower_query for w in ["child", "pediatric", "infant", "youth"]):
            demographics.append("Pediatric")
        if any(w in lower_query for w in ["female", "women", "male", "men", "gender"]):
            demographics.append("Gender-Specific")

        return {
            "original_query": user_query,
            "extracted_keywords": list(set(keywords)),
            "named_entities": entities,
            "inferred_demographics": demographics
        }

# --- QUICK LOCAL TEST RUN ---
if __name__ == "__main__":
    parser = ClinicalQueryParser()
    
    test_query = "Find cardiovascular metrics and body mass index tracking for adults over 50"
    parsed_output = parser.parse(test_query)
    
    print("\n🔍 --- SPACY QUERY PARSING ANALYSIS ---")
    print(f"Query: {parsed_output['original_query']}\n")
    print(f"🎯 Keywords for Search Expansion: {parsed_output['extracted_keywords']}")
    print(f"👥 Inferred Demographics: {parsed_output['inferred_demographics']}")
    print(f"🏛️ Extracted Entities:")
    for ent in parsed_output['named_entities']:
        print(f"  - [{ent['label']}] {ent['text']} -> ({ent['meaning']})")
    print("-" * 40)
    