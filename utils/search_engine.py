import pandas as pd
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import torch
import os

# ==============================================================================
# SECTION 1: CONSTANTS AND CONFIGURATION
# ==============================================================================

# --- File Paths ---
PROCESSED_DATA_CSV = "final_processed_data.csv"
EMBEDDINGS_FILE = "embeddings.npy"

# --- Model Configuration ---
MODEL_NAME = "vinai/phobert-base"

# --- Search Hyperparameters ---
TOP_K = 10                 # Number of final results to return
CANDIDATE_POOL_SIZE = 50   # Number of candidates to retrieve in the first stage
SCORE_THRESHOLD = 0.015    # Lower threshold as re-ranking will refine scores

# --- Re-ranking Weights ---
# How much to value the initial retrieval score vs. the semantic re-rank score
RETRIEVAL_WEIGHT = 0.4
RERANK_WEIGHT = 0.6

# ==============================================================================
# SECTION 2: THE ADVANCED SEARCH ENGINE CLASS
# ==============================================================================

class SearchEngine:
    """
    An advanced 2-stage hybrid search engine.
    Stage 1 (Retrieval): Quickly gathers a large pool of candidates using BM25 + FAISS.
    Stage 2 (Re-ranking): Intelligently re-ranks the candidates using pure semantic similarity.
    """
    def __init__(self):
        """Initializes the search engine by loading all necessary components."""
        print("🚀 Khởi tạo Search Engine (Kiến trúc 2 giai đoạn)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Sử dụng thiết bị: {self.device}")

        self._load_dependencies()
        self._build_indexes()
        
        print("✅ Search Engine đã sẵn sàng!")

    def _load_dependencies(self):
        """Loads the dataset, embeddings, tokenizer, and model."""
        print("💾 Đang tải dữ liệu và mô hình...")
        try:
            self.df = pd.read_csv(PROCESSED_DATA_CSV)
            self.embeddings = np.load(EMBEDDINGS_FILE).astype('float32')
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
            self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
            self.model.eval()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Lỗi: Không tìm thấy file cần thiết. {e}")

    def _build_indexes(self):
        """Builds the FAISS and BM25 search indexes for the retrieval stage."""
        # --- Build FAISS Index ---
        print("... ⚙️  Đang xây dựng FAISS index (Giai đoạn 1)...")
        d = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(d)
        if self.device.type == 'cuda':
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        self.faiss_index.add(self.embeddings)

        # --- Build BM25 Index ---
        print("... ⚙️  Đang xây dựng BM25 index (Giai đoạn 1)...")
        corpus = self.df['text_for_embedding'].fillna('').tolist()
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print("✅ Các chỉ mục Retrieval đã sẵn sàng.")

    def _encode_query(self, query_text):
        """Converts a text query into a PhoBERT embedding vector."""
        with torch.no_grad():
            inputs = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            outputs = self.model(**inputs)
            # Return tensor on GPU for faster calculations
            return outputs.last_hidden_state[:, 0, :]

    def search(self, query):
        """Performs a 2-stage hybrid search."""
        print(f"\n🔍 Đang tìm kiếm cho truy vấn: '{query}'")
        
        # ==================== STAGE 1: RETRIEVAL ====================
        print(f"    -> Giai đoạn 1: Thu thập {CANDIDATE_POOL_SIZE} ứng viên tiềm năng...")
        
        query_embedding_gpu = self._encode_query(query)
        query_embedding_cpu = query_embedding_gpu.cpu().numpy()
        
        # Semantic search candidates
        _, semantic_indices = self.faiss_index.search(query_embedding_cpu, CANDIDATE_POOL_SIZE)
        
        # Keyword search candidates
        bm25_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(bm25_tokens)
        keyword_indices = np.argsort(bm25_scores)[::-1][:CANDIDATE_POOL_SIZE]
        
        # Combine unique candidates
        candidate_indices = np.union1d(semantic_indices[0], keyword_indices)
        
        if len(candidate_indices) == 0:
            print("    -> Không tìm thấy kết quả nào.")
            return pd.DataFrame()

        # =================== STAGE 2: RE-RANKING ====================
        print(f"    -> Giai đoạn 2: Tái xếp hạng {len(candidate_indices)} ứng viên bằng điểm ngữ nghĩa...")

        # Get embeddings for all candidates
        candidate_embeddings = torch.from_numpy(self.embeddings[candidate_indices]).to(self.device)
        
        # Calculate pure semantic similarity (Cosine Similarity) on GPU
        # This is the "algorithmic compensation" you asked for
        query_norm = torch.nn.functional.normalize(query_embedding_gpu, p=2, dim=1)
        candidates_norm = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=1)
        rerank_scores = torch.mm(query_norm, candidates_norm.transpose(0, 1)).flatten().cpu().numpy()

        # Create a DataFrame for easy ranking
        rank_df = pd.DataFrame({
            'id': candidate_indices,
            'bm25': bm25_scores[candidate_indices],
            'rerank': rerank_scores
        })

        # Normalize scores to a 0-1 range for fair combination
        rank_df['bm25_norm'] = (rank_df['bm25'] - rank_df['bm25'].min()) / (rank_df['bm25'].max() - rank_df['bm25'].min() + 1e-6)
        
        # Combine scores
        rank_df['final_score'] = (RETRIEVAL_WEIGHT * rank_df['bm25_norm']) + (RERANK_WEIGHT * rank_df['rerank'])

        # Sort by the final combined score
        top_indices = rank_df.sort_values('final_score', ascending=False).head(TOP_K)['id'].values
        
        # --- Format Output ---
        results_df = self.df.iloc[top_indices].copy()
        # Join final scores for context
        final_scores_df = rank_df[rank_df['id'].isin(top_indices)].set_index('id')
        results_df = results_df.join(final_scores_df, on=self.df.iloc[top_indices].index)

        return results_df[['id', 'name', 'address', 'final_score']].rename(columns={'final_score': 'score'})


def main():
    """Initializes the engine and runs test queries."""
    try:
        engine = SearchEngine()
        
        test_queries = [
            "tôi muốn ăn xá xíu", 
            "quán ăn gia đình rộng rãi", 
            "nem nướng thanh vân",
            "com chay",
            "bánh mì kẹp thịt",
        ]
        
        for q in test_queries:
            search_results = engine.search(q)
            print("------ KẾT QUẢ ------")
            if not search_results.empty:
                print(search_results)
            print("="*20)

    except Exception as e:
        print(f"\n💥 Đã xảy ra lỗi nghiêm trọng: {e}")

if __name__ == '__main__':
    main()

