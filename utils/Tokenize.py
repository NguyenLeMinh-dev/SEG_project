import pandas as pd
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import os

# ==============================================================================
# SECTION 1: CONSTANTS AND CONFIGURATION
# ==============================================================================

# --- Input/Output Files ---
# Reads the final processed data and outputs PyTorch tensors.
INPUT_CSV = "final_processed_data.csv"
OUTPUT_TENSORS = "tokenized_data.pt"

# --- Tokenizer Configuration ---
MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 256  # Max sequence length for PhoBERT
BATCH_SIZE = 32   # Process 32 sentences at a time for efficiency

# ==============================================================================
# SECTION 2: MAIN PROCESSING FUNCTION
# ==============================================================================

def tokenize_for_phobert():
    """
    Main function to run the tokenization pipeline.
    """
    # --- 1. Load the clean dataset ---
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"📖 Đã đọc {len(df)} dòng từ '{INPUT_CSV}'.")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{INPUT_CSV}'. Vui lòng chạy script 'clean_data.py' trước.")
        return

    if 'text_for_embedding' not in df.columns:
        print(f"❌ Lỗi: Không tìm thấy cột 'text_for_embedding' trong file CSV.")
        return

    # --- 2. Load the PhoBERT tokenizer ---
    print(f"🤖 Đang tải PhoBERT tokenizer ('{MODEL_NAME}')...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        print("✅ Tải tokenizer thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi tải tokenizer: {e}. Vui lòng kiểm tra kết nối mạng.")
        return

    # --- 3. Tokenize the text in batches ---
    texts = df['text_for_embedding'].tolist()
    print(f"\n⚙️  Bắt đầu tokenize {len(texts)} dòng văn bản...")
    
    all_input_ids = []
    all_attention_masks = []

    # Use tqdm for a progress bar while iterating through batches
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Tokenizing batches"):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        tokenized_batch = tokenizer(
            batch_texts,
            padding='max_length',  # Pad all sentences to MAX_LENGTH
            truncation=True,       # Truncate sentences longer than MAX_LENGTH
            max_length=MAX_LENGTH,
            return_tensors='pt'    # Return PyTorch tensors
        )
        
        all_input_ids.append(tokenized_batch['input_ids'])
        all_attention_masks.append(tokenized_batch['attention_mask'])

    # Concatenate all batch tensors into single large tensors
    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_masks, dim=0)

    tokenized_output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    print("✅ Tokenize hoàn tất!")

    # --- 4. Save the results ---
    input_ids_shape = tokenized_output['input_ids'].shape
    print(f"\nKích thước của tensor 'input_ids': {input_ids_shape}")
    print(f"(Số lượng câu: {input_ids_shape[0]}, Độ dài tối đa: {input_ids_shape[1]})")

    try:
        torch.save(tokenized_output, OUTPUT_TENSORS)
        print(f"\n💾 Đã lưu kết quả vào file: '{OUTPUT_TENSORS}'")
        print("👉 Bước tiếp theo: Dùng file này để chạy 'generate_embeddings.py'.")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")

# ==============================================================================
# SECTION 3: SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    tokenize_for_phobert()
