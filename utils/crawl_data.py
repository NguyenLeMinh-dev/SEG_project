from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import csv

# Service Edge Driver
service = Service(r"C:\Program Files (x86)\msedgedriver.exe")
driver = webdriver.Edge(service=service)

# Link crawl
start_url = "https://www.foody.vn/can-tho"
driver.get(start_url)
time.sleep(3)

# Cuộn xuống để load thêm dữ liệu
for _ in range(5):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

elements = driver.find_elements(By.TAG_NAME, "a")
def get_links(elements):
    links = set()
    for el in elements:
        href = el.get_attribute("href")
        if not href:
            continue
        href = href.strip()
        if (
            "foody.vn/can-tho" in href
            and not any(x in href for x in ["#", "/food/", "/su-kien", "/bo-suu-tap", "/bai-viet", "/video", "/khuyen-mai", "/coupon", "/o-dau","/hinh-anh", "/top-thanh-vien"])
        ):
            clean = re.sub(r"/binh-luan-\d+", "", href)
            clean = clean.split("/binh-luan")[0]
            links.add(clean)

    return list(set(links))

links = get_links(elements)
print(f"\nTotal link: {len(links)}")


data = []
def safe_find(driver, selector, attr="text", timeout=2):
    """Tìm phần tử an toàn, trả về chuỗi rỗng nếu không thấy."""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return element.get_attribute(attr) if attr != "text" else element.text.strip()
    except:
        return ""

data = []

for i, url in enumerate(links[:5], 1):  # 👈 Giới hạn 5 link đầu để test
    print(f"\n[{i}/{len(links)}] Đang crawl: {url}")
    driver.get(url)
    time.sleep(2)  # nhỏ lại vì đã có WebDriverWait hỗ trợ

    # Lấy dữ liệu bằng hàm safe_find
    name = safe_find(driver, "h1[itemprop='name']")
    address = safe_find(driver, 'span[itemprop="streetAddress"]')
    district = safe_find(driver, 'span[itemprop="addressLocality"]')
    region = safe_find(driver, 'span[itemprop="addressRegion"]')
    rating = safe_find(driver, "div[itemprop='ratingValue']")
    image = safe_find(driver, 'img[itemprop="image"]', attr="src")
    lon = safe_find(driver, 'meta[property="place:location:longitude"]', attr="content")
    lat = safe_find(driver, 'meta[property="place:location:latitude"]', attr="content")
    price = safe_find(driver, 'span[itemprop="priceRange"]', attr="innerText")
    hours_text = safe_find(driver, '.micro-timesopen span:nth-of-type(3)', attr="innerText")

    # Xử lý dữ liệu
    gps = ", ".join(x for x in [lat, lon] if x)
    address_full = ", ".join(x for x in [address, district, region] if x)

    # Thêm vào danh sách kết quả
    data.append({
        "name": name,
        "address": address_full,
        "rating": rating,
        "open_close": hours_text,
        "price": price,
        "url": url,
        "gps": gps,
        "image_src": image
    })


# =============================
# 4️⃣  LƯU KẾT QUẢ RA FILE CSV
# =============================
output_file = "foody_cantho.csv"
with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

print(f"\n✅ Đã lưu {len(data)} dòng dữ liệu vào file: {output_file}")

# =============================
driver.quit()
