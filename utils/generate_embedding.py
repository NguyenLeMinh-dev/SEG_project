import torch
from transformers import AutoModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os

# ==============================================================================
# SECTION 1: CONSTANTS AND CONFIGURATION
# ==============================================================================

# --- Input/Output Files ---
INPUT_TENSORS = "tokenized_data.pt"
OUTPUT_EMBEDDINGS = "embeddings.npy"

# --- Model & Hardware Configuration ---
MODEL_NAME = "vinai/phobert-base"
# Increase batch size for powerful GPUs like the 4060 Ti to maximize performance
BATCH_SIZE = 128 

# ==============================================================================
# SECTION 2: MAIN PROCESSING FUNCTION
# ==============================================================================

def generate_embeddings():
    """
    Main function to run the embedding generation pipeline.
    This is the most computationally intensive step.
    """
    # --- 1. Setup device (GPU/CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Sử dụng thiết bị: {device}")
    if device.type == 'cpu':
        print("⚠️  Cảnh báo: Chạy trên CPU sẽ chậm hơn đáng kể. Hãy cân nhắc dùng Google Colab.")

    # --- 2. Load tokenized data ---
    try:
        tokenized_data = torch.load(INPUT_TENSORS)
        dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"💾 Đã tải dữ liệu đã tokenize từ '{INPUT_TENSORS}'.")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{INPUT_TENSORS}'. Vui lòng chạy 'Tokenize.py' trước.")
        return

    # --- 3. Load the PhoBERT model ---
    print(f"🤖 Đang tải mô hình PhoBERT ('{MODEL_NAME}')...")
    try:
        # Move model to the selected device (GPU)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        # Set to evaluation mode to disable dropout and save memory
        model.eval()
        print("✅ Tải mô hình thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}. Vui lòng kiểm tra kết nối mạng.")
        return
        
    # --- 4. Generate Embeddings in Batches ---
    all_embeddings = []
    print(f"\n⚙️  Bắt đầu tạo embeddings với batch size = {BATCH_SIZE}...")
    
    # Disable gradient calculation to save VRAM and speed up inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            # Move data batch to the GPU
            b_input_ids, b_attention_mask = [b.to(device) for b in batch]
            
            # Get model outputs
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
            
            # Extract the [CLS] token's embedding, which represents the entire sentence
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Move embeddings back to CPU and convert to NumPy array
            all_embeddings.append(cls_embeddings.cpu().numpy())

    # Combine embeddings from all batches into a single NumPy matrix
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    print("✅ Tạo embeddings hoàn tất!")
    print(f"Kích thước của ma trận embeddings: {final_embeddings.shape}")

    # --- 5. Save the final embeddings ---
    try:
        np.save(OUTPUT_EMBEDDINGS, final_embeddings)
        print(f"\n💾 Đã lưu embeddings vào file: '{OUTPUT_EMBEDDINGS}'")
        print("👉 Bước tiếp theo: Dùng file này để chạy 'search_engine.py'.")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")

# ==============================================================================
# SECTION 3: SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    generate_embeddings()
