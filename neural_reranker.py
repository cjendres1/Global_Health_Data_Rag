import logging
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralReranker")


class PyTorchNeuralReranker:
    """
    Cross-encoder reranker used as the second stage of retrieval.

    The model receives:
        (user query, candidate metadata)

    and produces a relevance score for each pair.
    """

    def __init__(self):
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # Select GPU when available, otherwise CPU.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(
            "Loading Neural Reranker: %s on %s",
            self.model_name,
            self.device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(self.model_name)
            .to(self.device)
        )

        self.model.eval()

        logger.info("Neural reranker ready.")

    def rerank(
        self,
        query: str,
        candidate_metadata_list: list,
        return_timing: bool = False,
    ):
        """
        Rerank candidate metadata using the cross-encoder.

        Parameters
        ----------
        query : str
            Original user query.

        candidate_metadata_list : list
            Candidate metadata dictionaries returned from ChromaDB.

        return_timing : bool
            If True, return both results and timing information.

        Returns
        -------
        list
            Reranked candidates.

        OR

        tuple[list, dict]
            Reranked candidates and timing metrics.
        """

        if not candidate_metadata_list:
            if return_timing:
                return [], {
                    "tokenization_seconds": 0.0,
                    "inference_seconds": 0.0,
                    "total_seconds": 0.0,
                    "candidate_count": 0,
                }

            return []

        start_total = time.perf_counter()

        # ---------------------------------------------------------------------
        # Construct query/document pairs
        # ---------------------------------------------------------------------
        pairs = [
            [
                query,
                (
                    f"{meta.get('source_dataset', 'Unknown')} | "
                    f"{meta.get('variable_name', 'Unknown')} | "
                    f"{meta.get('description', '')}"
                ),
            ]
            for meta in candidate_metadata_list
        ]

        # ---------------------------------------------------------------------
        # Tokenization
        # ---------------------------------------------------------------------
        start_tokenization = time.perf_counter()

        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        tokenization_seconds = time.perf_counter() - start_tokenization

        # ---------------------------------------------------------------------
        # Neural inference
        # ---------------------------------------------------------------------
        start_inference = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(**features)

        inference_seconds = time.perf_counter() - start_inference

        # ---------------------------------------------------------------------
        # Convert raw logits to bounded 0-1 display score
        #
        # IMPORTANT:
        # This is NOT a calibrated probability.
        # It is simply a normalized relevance score.
        # ---------------------------------------------------------------------
        raw_scores = outputs.logits.squeeze(-1)

        normalized_scores = torch.sigmoid(raw_scores)

        scores = normalized_scores.detach().cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        # ---------------------------------------------------------------------
        # Attach scores
        # ---------------------------------------------------------------------
        for idx, score in enumerate(scores):
            candidate_metadata_list[idx]["rerank_score"] = float(score)

        # ---------------------------------------------------------------------
        # Sort by descending neural relevance
        # ---------------------------------------------------------------------
        sorted_candidates = sorted(
            candidate_metadata_list,
            key=lambda x: x.get("rerank_score", 0.0),
            reverse=True,
        )

        total_seconds = time.perf_counter() - start_total

        timing = {
            "tokenization_seconds": tokenization_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": total_seconds,
            "candidate_count": len(candidate_metadata_list),
        }

        if return_timing:
            return sorted_candidates, timing

        return sorted_candidates


# -----------------------------------------------------------------------------
# LOCAL TEST
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    reranker = PyTorchNeuralReranker()

    user_query = "body mass index metrics"

    mock_chroma_candidates = [
        {
            "source_dataset": "NHANES",
            "variable_name": "RIAGENDR",
            "description": (
                "Gender of the participant. 1=Male, 2=Female."
            ),
        },
        {
            "source_dataset": "UK_Biobank",
            "variable_name": "f.21001",
            "description": (
                "Anthropometric measurement of weight divided "
                "by squared height (BMI)."
            ),
        },
    ]

    results, timing = reranker.rerank(
        user_query,
        mock_chroma_candidates,
        return_timing=True,
    )

    print("\n--- NEURAL RERANKER TEST ---")

    for doc in results:
        print(
            f"Dataset: {doc['source_dataset']} | "
            f"Var: {doc['variable_name']}"
        )
        print(
            f"Neural Relevance: "
            f"{doc['rerank_score']:.4f}"
        )
        print("-" * 50)

    print("\nTiming:")
    for key, value in timing.items():
        print(f"{key}: {value}")
        