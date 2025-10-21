import os
import json
from flask import Flask, request, jsonify, send_from_directory # <--- THÊM 'send_from_directory'
from flask_cors import CORS
from utils.search_engine import SearchEngine
import pandas as pd

# Cấu hình đường dẫn đến thư mục chứa ảnh
# Giả sử thư mục 'food_images' nằm cùng cấp với 'app.py'
# Giả sử thư mục dự án của bạn là ~/Documents/SEG_project/
# và app.py nằm trong đó.
# Đường dẫn này sẽ trỏ chính xác đến ~/Documents/SEG_project/food_images/
IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'food_images')

app = Flask(__name__)
CORS(app)

print("🚀 Đang khởi tạo Search Engine... Vui lòng đợi.")
try:
    engine = SearchEngine()
    print("✅ Search Engine đã sẵn sàng nhận yêu cầu.")
except FileNotFoundError as e:
    print(f"💥 LỖI NGHIÊM TRỌNG: {e}")
    engine = None

# ==============================================================================
# ROUTE MỚI: ĐỂ PHỤC VỤ HÌNH ẢNH
# ==============================================================================
@app.route('/images/<path:filename>')
def get_image(filename):
    """
    Phục vụ file ảnh tĩnh từ thư mục 'food_images'.
    """
    print(f"Đang phục vụ ảnh: {filename}")
    return send_from_directory(IMAGE_FOLDER, filename)

# ==============================================================================
# ROUTE CŨ: ĐỂ TÌM KIẾM
# ==============================================================================
@app.route('/search', methods=['GET'])
def search_api():
    # === THÊM CÁC BIỂN BÁO DEBUG ===
    print("\n\n=======================================")
    print(f"✅ [app.py] ĐÃ NHẬN ĐƯỢC YÊU CẦU: {request.url}")

    if not engine:
        print("❌ [app.py] LỖI: Engine chưa sẵn sàng.")
        return jsonify({"error": "Search engine chưa được khởi tạo."}), 500

    query = request.args.get('q', '')
    if not query:
        print("❌ [app.py] LỖI: Không có query.")
        return jsonify({"error": "Vui lòng cung cấp query (tham số 'q')."}), 400

    try:
        print(f"🚀 [app.py] BẮT ĐẦU GỌI engine.search(query='{query}') ...")
        results_df = engine.search(query)
        print(f"✅ [app.py] GỌI engine.search() THÀNH CÔNG.")
        
        if results_df.empty:
            print("🟡 [app.py] Kết quả rỗng.")
            return jsonify([])

        results_json = results_df.to_dict('records')
        print(f"✅ [app.py] Đang gửi {len(results_json)} kết quả về trình duyệt.")
        return jsonify(results_json)

    except Exception as e:
        print(f"💥💥💥 [app.py] LỖI NGHIÊM TRỌNG TRONG KHI TÌM KIẾM: {e}")
        return jsonify({"error": "Đã xảy ra lỗi máy chủ nội bộ."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)