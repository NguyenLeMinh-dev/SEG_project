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
# Đường dẫn này là tương đối so với nơi bạn chạy 'app.py'
# Giả sử 'final_processed_data.csv' và 'embeddings.npy' nằm cùng cấp với 'app.py'
PROCESSED_DATA_CSV = "final_processed_data.csv"
EMBEDDINGS_FILE = "embeddings.npy"

# --- Model Configuration ---
MODEL_NAME = "vinai/phobert-base"

# --- Search Hyperparameters ---
TOP_K = 10
CANDIDATE_POOL_SIZE = 50
SCORE_THRESHOLD = 0.015

# --- Re-ranking Weights ---
RETRIEVAL_WEIGHT = 0.4
RERANK_WEIGHT = 0.6

# --- Image URL Base ---
# URL này phải khớp với cách 'app.py' chạy
BASE_IMAGE_URL = "http://127.0.0.1:5000/images/"


# ==============================================================================
# SECTION 2: THE ADVANCED SEARCH ENGINE CLASS
# ==============================================================================

class SearchEngine:
    def __init__(self):
        print("🚀 Khởi tạo Search Engine (Kiến trúc 2 giai đoạn)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Sử dụng thiết bị: {self.device}")

        self._load_dependencies()
        self._build_indexes()
        
        print("✅ Search Engine đã sẵn sàng!")

    def _load_dependencies(self):
        print("💾 Đang tải dữ liệu và mô hình...")
        if not os.path.exists(PROCESSED_DATA_CSV):
            raise FileNotFoundError(f"Lỗi: Không tìm thấy file '{PROCESSED_DATA_CSV}'. Đảm bảo nó nằm cùng cấp với app.py.")
        if not os.path.exists(EMBEDDINGS_FILE):
             raise FileNotFoundError(f"Lỗi: Không tìm thấy file '{EMBEDDINGS_FILE}'. Đảm bảo nó nằm cùng cấp với app.py.")

        self.df = pd.read_csv(PROCESSED_DATA_CSV)
        self.embeddings = np.load(EMBEDDINGS_FILE).astype('float32')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    def _build_indexes(self):
        print("... ⚙️  Đang xây dựng FAISS index (Giai đoạn 1)...")
        d = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(d)
        if self.device.type == 'cuda':
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        self.faiss_index.add(self.embeddings)

        print("... ⚙️  Đang xây dựng BM25 index (Giai đoạn 1)...")
        corpus = self.df['text_for_embedding'].fillna('').tolist()
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print("✅ Các chỉ mục Retrieval đã sẵn sàng.")

    def _encode_query(self, query_text):
        with torch.no_grad():
            inputs = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :]

    def search(self, query):
        print(f"\n🔍 Đang tìm kiếm cho truy vấn: '{query}'")
        
        # STAGE 1: RETRIEVAL
        print(f"    -> Giai đoạn 1: Thu thập ứng viên...")
        query_embedding_gpu = self._encode_query(query)
        query_embedding_cpu = query_embedding_gpu.cpu().numpy()
        
        _, semantic_indices = self.faiss_index.search(query_embedding_cpu, CANDIDATE_POOL_SIZE)
        
        bm25_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(bm25_tokens)
        keyword_indices = np.argsort(bm25_scores)[::-1][:CANDIDATE_POOL_SIZE]
        
        candidate_indices = np.union1d(semantic_indices[0], keyword_indices)
        
        if len(candidate_indices) == 0:
            return pd.DataFrame()

        # STAGE 2: RE-RANKING
        print(f"    -> Giai đoạn 2: Tái xếp hạng {len(candidate_indices)} ứng viên...")
        candidate_embeddings = torch.from_numpy(self.embeddings[candidate_indices]).to(self.device)
        
        query_norm = torch.nn.functional.normalize(query_embedding_gpu, p=2, dim=1)
        candidates_norm = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=1)
        rerank_scores = torch.mm(query_norm, candidates_norm.transpose(0, 1)).flatten().cpu().numpy()

        rank_df = pd.DataFrame({
            'id': candidate_indices,
            'bm25': bm25_scores[candidate_indices],
            'rerank': rerank_scores
        })

        rank_df['bm25_norm'] = (rank_df['bm25'] - rank_df['bm25'].min()) / (rank_df['bm25'].max() - rank_df['bm25'].min() + 1e-6)
        rank_df['final_score'] = (RETRIEVAL_WEIGHT * rank_df['bm25_norm']) + (RERANK_WEIGHT * rank_df['rerank'])

        top_indices = rank_df.sort_values('final_score', ascending=False).head(TOP_K)['id'].values
        
        # --- Format Output ---
        results_df = self.df.iloc[top_indices].copy()
        final_scores_df = rank_df[rank_df['id'].isin(top_indices)].set_index('id')
        
        # === SỬA LỖI 1: SỬA LỖI .JOIN() GÂY TREO ===
        # Xóa 'on=...' để join dựa trên index
        results_df = results_df.join(final_scores_df) 

        # === SỬA LỖI 2: TẠO CỘT MỚI CHO FRONTEND ===
        
        # 2a. Tạo cột 'gps' từ 'gps_lat' và 'gps_long'
        if 'gps_lat' in results_df.columns and 'gps_long' in results_df.columns:
            results_df['gps'] = results_df['gps_lat'].astype(str) + ',' + results_df['gps_long'].astype(str)
            results_df['gps'] = results_df['gps'].replace('nan,nan', np.nan)
        else:
            print("CẢNH BÁO: Không tìm thấy cột 'gps_lat' hoặc 'gps_long'.")
            results_df['gps'] = np.nan

        # 2b. Tạo cột 'image_src' (URL đầy đủ) từ 'image_path'
        # ...
        if 'image_path' in results_df.columns:
            results_df['image_src'] = results_df['image_path'].apply(
                # os.path.basename sẽ lấy ra phần cuối cùng của đường dẫn (tên file)
                # Ví dụ: 'food_images/000641.jpg' -> '000641.jpg'
                lambda x: f"{BASE_IMAGE_URL}{os.path.basename(x)}" if pd.notna(x) else np.nan
            )
      
        else:
            print("CẢNH BÁO: Không tìm thấy cột 'image_path'.")
            results_df['image_src'] = np.nan

        # --- Lấy các cột cuối cùng mà frontend cần ---
        required_cols = ['id', 'name', 'address', 'gps', 'image_src', 'final_score']
        
        available_cols = [col for col in required_cols if col in results_df.columns]
        
        return results_df[available_cols].rename(columns={'final_score': 'score'})

# (Hàm main() dùng để test, giữ nguyên)
def main():
    try:
        engine = SearchEngine()
        test_queries = ["bánh mì", "com chay"]
        for q in test_queries:
            search_results = engine.search(q)
            print("------ KẾT QUẢ ------")
            if not search_results.empty:
                print(search_results[['name', 'gps', 'image_src', 'score']]) # In ra các cột mới
            print("="*20)
    except Exception as e:
        print(f"\n💥 Đã xảy ra lỗi nghiêm trọng: {e}")

if __name__ == '__main__':
    main()