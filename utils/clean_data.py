import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import re
import os
from PIL import Image
from io import BytesIO
import unidecode
import concurrent.futures

# ==============================================================================
# SECTION 1: CONSTANTS AND CONFIGURATION
# ==============================================================================

# --- Input/Output Files ---
INPUT_CSV = "foody_cantho_with_tags.csv"
OUTPUT_CSV = "final_processed_data.csv"
IMAGE_FOLDER = "food_images"
COMMENT_CHAR_LIMIT = 450 # Giới hạn ký tự cho phần bình luận

# ==============================================================================
# SECTION 2: HELPER FUNCTIONS FOR DATA CLEANING
# ==============================================================================

def clean_text(text):
    """General text cleaning: removes extra whitespace and unwanted characters."""
    if pd.isna(text):
        return ''
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\sÀ-ỹ,.-]', '', text)
    return text

def clean_comment_text(text):
    """Specific cleaning for comments, preserving more punctuation."""
    if pd.isna(text):
        return ''
    text = str(text).strip().replace('\n', '. ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\sÀ-ỹ.,!?]', '', text)
    return text

def remove_accents(text):
    """Converts Vietnamese text to lowercase, unaccented text."""
    return unidecode.unidecode(str(text)).lower().strip()

def clean_price(price_str):
    """Extracts min and max price from a string."""
    if pd.isna(price_str):
        return None, None
    s = str(price_str).replace('đ', '').replace('.', '').replace(',', '').strip()
    numbers = re.findall(r'\d+', s)
    if len(numbers) == 1:
        val = float(numbers[0])
        return val, val
    elif len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    return None, None

def clean_rating(r):
    """Standardizes rating scores to a 0-10 scale."""
    try:
        val = float(r)
        return round(val, 2) if val <= 10 else round(val / 10, 2)
    except (ValueError, TypeError):
        return None

def clean_open_close(t):
    """Extracts opening and closing hours from a string."""
    times = re.findall(r'\d{1,2}:\d{2}', str(t))
    if len(times) >= 2:
        try:
            return float(times[0].split(':')[0]), float(times[1].split(':')[0])
        except (ValueError, IndexError):
            return None, None
    return None, None

def clean_gps(gps_str):
    """Extracts latitude and longitude from a string."""
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(gps_str))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None, None

def download_image_worker(args):
    """Worker function to download and resize a single image."""
    url, folder, size, img_id = args
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img = img.resize(size)
            filename = f"{img_id:06d}.jpg"
            path = os.path.join(folder, filename)
            img.save(path, "JPEG", quality=90)
            return path.replace("\\", "/")
    except Exception:
        return None
    return None

# ==============================================================================
# SECTION 3: MAIN PROCESSING FUNCTION
# ==============================================================================

def main():
    """
    Main function to run the entire data cleaning and processing pipeline.
    """
    # --- 1. Read the source CSV file ---
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"📖 Đã đọc {len(df)} dòng từ '{INPUT_CSV}'.")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{INPUT_CSV}'. Vui lòng chạy script 'patch_tags_scraper.py' trước.")
        return

    # --- 2. Clean and Standardize Data ---
    print("✨ Bắt đầu làm sạch và chuẩn hóa dữ liệu...")
    df['name'] = df['name'].apply(clean_text)
    df['address'] = df['address'].apply(clean_text)
    df['rating'] = df['rating'].apply(clean_rating)
    df['comments'] = df['comments'].apply(clean_comment_text)
    df['tags'] = df['tags'].apply(clean_text)

    df[['price_min', 'price_max']] = df['price'].apply(lambda x: pd.Series(clean_price(x)))
    df[['open_hour', 'close_hour']] = df['open_close'].apply(lambda x: pd.Series(clean_open_close(x)))
    df[['gps_lat', 'gps_long']] = df['gps'].apply(lambda x: pd.Series(clean_gps(x)))

    # --- 3. Download Images in Parallel ---
    df.dropna(subset=['name', 'image_src'], inplace=True)
    df = df.reset_index(drop=True)
    
    print(f"🖼️  Đang tải {len(df)} ảnh (chạy song song)...")
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    tasks = [(row['image_src'], IMAGE_FOLDER, (224, 224), i + 1) for i, row in df.iterrows()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(download_image_worker, tasks), total=len(tasks), desc="Downloading images"))
    df['image_path'] = results
    
    df.dropna(subset=['image_path'], inplace=True)
    df = df.reset_index(drop=True)
    print(f"✅ Đã tải thành công {len(df)} ảnh.")

    # --- 4. Create Enriched Text for Embedding ---
    print("✍️  Tạo cột 'text_for_embedding' giàu ngữ cảnh...")
    df['name_no_accent'] = df['name'].apply(remove_accents)
    
    def extract_district(address):
        match = re.search(r'Quận\s+([\w\s]+)', str(address), re.IGNORECASE)
        return match.group(1).strip() if match else 'Khác'
    df['district'] = df['address'].apply(extract_district)
    df['city'] = 'Cần Thơ'

    def create_embedding_text(row):
        """
        NÂNG CẤP: Kết hợp thông tin một cách thông minh và giới hạn độ dài
        để phù hợp với giới hạn token của PhoBERT.
        """
        parts = []
        
        # Ưu tiên 1: Tên quán
        parts.append(f"{row['name']}.")
        
        # Ưu tiên 2: Tags - tín hiệu ngữ nghĩa rõ ràng nhất
        if pd.notna(row['tags']) and row['tags']:
            parts.append(f"Thể loại: {row['tags']}.")
        
        # Ưu tiên 3: Bình luận (đã được rút gọn)
        if pd.notna(row['comments']) and row['comments']:
            # Cắt bớt bình luận để không quá dài
            truncated_comments = row['comments'][:COMMENT_CHAR_LIMIT]
            if len(row['comments']) > COMMENT_CHAR_LIMIT:
                truncated_comments += "..." # Thêm dấu ... nếu bình luận bị cắt
            parts.append(f"Một số đánh giá: {truncated_comments}")

        # Thông tin phụ: Địa chỉ
        parts.append(f"Địa chỉ tại {row['address']}.")
        
        return ' '.join(parts)
        
    df['text_for_embedding'] = df.apply(create_embedding_text, axis=1)

    # --- 5. Finalize and Save ---
    df['id'] = [f"{i:06d}" for i in range(1, len(df) + 1)]
    final_cols = [
        'id', 'name', 'name_no_accent', 'tags', 'address', 'district', 'city', 'rating',
        'price_min', 'price_max', 'open_hour', 'close_hour', 'gps_lat', 'gps_long',
        'text_for_embedding', 'image_path', 'comments', 'url'
    ]
    
    # Đảm bảo tất cả các cột đều tồn tại trước khi chọn
    existing_cols = [col for col in final_cols if col in df.columns]
    df_final = df[existing_cols]

    df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎉 HOÀN TẤT! {len(df_final)} dòng dữ liệu sạch đã được lưu tại '{OUTPUT_CSV}'")

# ==============================================================================
# SECTION 4: SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()

