import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralReranker")

class PyTorchNeuralReranker:
    def __init__(self):
        # Using a highly-optimized cross-encoder model checkpoint
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        # Check for CUDA availability to show GPU capabilities if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Neural Reranker to execution hardware device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval() # Put PyTorch layers into inference evaluation mode

    def rerank(self, query: str, candidate_metadata_list: list) -> list:
        """
        Takes an incoming query string and a list of structural variable metadata dictionary targets, 
        concatenates them, and executes a forward pass through the network to generate exact matching scores.
        """
        if not candidate_metadata_list:
            return []

        # Construct textual pairs for the Cross-Encoder architecture: (Query, Document Context)
        pairs = [
            [query, f"{meta['source_dataset']} | {meta['variable_name']} | {meta['description']}"] 
            for meta in candidate_metadata_list
        ]

        # Tokenize batch inputs directly onto our processing device
        features = self.tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        # Disable gradient calculations for highly optimized runtime execution performance
        with torch.no_grad():
            outputs = self.model(**features)
            # Cross-encoder logits indicate relative contextual matching performance alignment
            scores = outputs.logits.squeeze(-1).cpu().tolist()
            
            # If there's only one item, squeeze can collapse dimensions, handle edge cases safely
            if isinstance(scores, float):
                scores = [scores]

        # Attach scores back to metadata layers
        for idx, score in enumerate(scores):
            candidate_metadata_list[idx]["rerank_score"] = score

        # Sort elements downward on descending prediction score metrics
        sorted_candidates = sorted(candidate_metadata_list, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_candidates

# --- QUICK LOCAL INFRASTRUCTURE TEST ---
if __name__ == "__main__":
    reranker = PyTorchNeuralReranker()
    
    user_query = "body mass index metrics"
    mock_chroma_candidates = [
        {
            "source_dataset": "NHANES", 
            "variable_name": "RIAGENDR", 
            "description": "Gender of the participant. 1=Male, 2=Female."
        },
        {
            "source_dataset": "UK_Biobank", 
            "variable_name": "f.21001", 
            "description": "Anthropometric measurement of weight divided by squared height (BMI)."
        }
    ]
    
    results = reranker.rerank(user_query, mock_chroma_candidates)
    
    print("\n🔥 --- NEURAL RERANKER SCORE VALIDATION ---")
    for doc in results:
        print(f"Dataset: {doc['source_dataset']} | Var: {doc['variable_name']}")
        print(f"🤖 PyTorch Reranker Prediction Alignment Score: {doc['rerank_score']:.4f}")
        print("-" * 50)
        