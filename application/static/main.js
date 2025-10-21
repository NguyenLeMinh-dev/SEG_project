// --- BIẾN TOÀN CỤC & CẤU HÌNH ---
let userLocation = null;
let map = null;
let routingControl = null;
let animatedMarker = null;
let currentTravelMode = 'car';
let currentDestination = null;

const ICONS_SVG = {
    car: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-8 h-8 text-yellow-500 drop-shadow-lg"><path d="M5.507 8.493l-.434 2.598A3.75 3.75 0 008.25 15h7.5a3.75 3.75 0 003.178-3.909l-.434-2.598a.75.75 0 00-.73-.643H6.237a.75.75 0 00-.73.643zM12 3a.75.75 0 00-.75.75v.755a3 3 0 00-1.652.89l-.421-.422a.75.75 0 10-1.06 1.06l.421.422a3 3 0 00-.89 1.652H6.75a.75.75 0 00-.75.75v1.5c0 .414.336.75.75.75h.755a3 3 0 00.89 1.652l-.422.421a.75.75 0 101.06 1.06l.422-.421a3 3 0 001.652.89v.755a.75.75 0 001.5 0v-.755a3 3 0 001.652-.89l.421.422a.75.75 0 101.06-1.06l-.421-.422a3 3 0 00.89-1.652h.755a.75.75 0 00.75-.75v-1.5a.75.75 0 00-.75-.75h-.755a3 3 0 00-.89-1.652l.422-.421a.75.75 0 10-1.06-1.06l-.422.421a3 3 0 00-1.652-.89V3.75A.75.75 0 0012 3zM12 7.5a1.5 1.5 0 100 3 1.5 1.5 0 000-3z" /></svg>`,
    bike: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-8 h-8 text-gray-800"><path fill-rule="evenodd" d="M9.164 1.832a.75.75 0 01.515.243l3.375 4.125a.75.75 0 01-.243 1.031l-.478.359a.75.75 0 01-.986-.145l-2.03-3.248a.75.75 0 00-1.295.808l2.585 4.137a.75.75 0 01-.33 1.02l-.478.358a.75.75 0 01-.986-.145L7.5 7.152v2.1a.75.75 0 01-1.5 0v-3.41a.75.75 0 01.243-1.031l3.375-4.125a.75.75 0 01.546-.243zM14.5 2.25a2 2 0 100 4 2 2 0 000-4z" clip-rule="evenodd" /><path d="M11.25 11.25a.75.75 0 01.75-.75h1.5a.75.75 0 01.75.75v2.25a.75.75 0 01-1.5 0v-1.5h-.75a.75.75 0 01-.75-.75z" /><path d="M3.75 13.5a.75.75 0 000 1.5h10.536l-1.34 2.233a.75.75 0 101.248.746l2.122-3.536a.75.75 0 000-.746l-2.122-3.536a.75.75 0 10-1.248.746L14.286 15H3.75z" /><path d="M15.5 12.25a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zM4 12.25a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5z" /></svg>`,
    walk: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-8 h-8 text-gray-800"><path fill-rule="evenodd" d="M11.47 2.47a.75.75 0 011.06 0l4.5 4.5a.75.75 0 01-1.06 1.06l-3.22-3.22V16.5a.75.75 0 01-1.5 0V4.81L8.03 8.03a.75.75 0 01-1.06-1.06l4.5-4.5zM12 18a2.25 2.25 0 100 4.5 2.25 2.25 0 000-4.5z" clip-rule="evenodd" /><path d="M6.75 19.5a.75.75 0 00-1.5 0v2.25a.75.75 0 001.5 0V19.5zM3 19.5a.75.75 0 00-1.5 0v2.25a.75.75 0 001.5 0V19.5zM18.75 19.5a.75.75 0 00-1.5 0v2.25a.75.75 0 001.5 0V19.5zM15 19.5a.75.75 0 00-1.5 0v2.25a.75.75 0 001.5 0V19.5z" /></svg>`
};
// --- NÂNG CẤP: Thêm tốc độ trung bình (km/h) ---
const CUSTOM_SPEEDS_KMH = {
    car: 30,
    bike: 15,
    walk: 4
};
const TRAVEL_MODES = [
    { id: 'car', label: 'Xe hơi', icon: ICONS_SVG.car, profile: 'driving' },
    { id: 'bike', label: 'Xe đạp', icon: ICONS_SVG.bike, profile: 'bicycle' },
    { id: 'walk', label: 'Đi bộ', icon: ICONS_SVG.walk, profile: 'foot' }
];

// --- LẤY CÁC THÀNH PHẦN DOM ---
const getLocationBtn = document.getElementById('getLocationBtn');
const locationStatus = document.getElementById('locationStatus');
const mapModal = document.getElementById('mapModal');
const closeMapModalBtn = document.getElementById('closeMapModal');
const mapModalTitle = document.getElementById('mapModalTitle');
const startAnimationBtn = document.getElementById('startAnimationBtn');
const summaryDistance = document.getElementById('summary-distance');
const summaryTime = document.getElementById('summary-time');
const mapLoader = document.getElementById('map-loader');
const travelModeSelector = document.getElementById('travel-mode-selector');

const API_URL = "http://127.0.0.1:5000"; // Địa chỉ máy chủ Python của bạn
const restaurantListEl = document.getElementById('restaurant-list');
const searchInput = document.getElementById('search-input');
const searchButton = document.getElementById('search-button');
const searchStatus = document.getElementById('search-status');

// --- HÀM XỬ LÝ VỊ TRÍ ---
getLocationBtn.addEventListener('click', () => {
     if ("geolocation" in navigator) {
        locationStatus.textContent = "Đang xác định vị trí...";
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                userLocation = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                locationStatus.textContent = `Vị trí của bạn: ${userLocation.lat.toFixed(5)}, ${userLocation.lng.toFixed(5)}`;
                locationStatus.style.color = 'green';
            },
            (err) => {
                userLocation = null;
                locationStatus.textContent = "Lỗi: Không thể lấy vị trí của bạn.";
                locationStatus.style.color = 'red';
            }
        );
    }
});

// --- HÀM TÌM KIẾM ---

// Hàm gọi API và hiển thị kết quả
async function performSearch() {
    const query = searchInput.value.trim();
    if (query === "") {
        searchStatus.textContent = "Vui lòng nhập từ khóa tìm kiếm.";
        searchStatus.style.color = 'red';
        return;
    }

    searchStatus.textContent = "Đang tìm kiếm...";
    searchStatus.style.color = 'gray';
    restaurantListEl.innerHTML = ''; // Xóa kết quả cũ

    try {
        const response = await fetch(`${API_URL}/search?q=${encodeURIComponent(query)}`);
        
        if (!response.ok) {
            throw new Error(`Lỗi máy chủ: ${response.statusText}`);
        }

        const results = await response.json();
        
        if (results.length === 0) {
            searchStatus.textContent = "Không tìm thấy kết quả nào phù hợp.";
            searchStatus.style.color = 'gray';
        } else {
            searchStatus.textContent = `Tìm thấy ${results.length} kết quả.`;
            searchStatus.style.color = 'green';
            renderRestaurantList(results);
        }

    } catch (error) {
        console.error("Lỗi khi gọi API:", error);
        searchStatus.textContent = "Lỗi: Không thể kết nối đến máy chủ tìm kiếm.";
        searchStatus.style.color = 'red';
    }
}

// Gắn sự kiện cho nút tìm kiếm
searchButton.addEventListener('click', performSearch);
// Cho phép nhấn Enter để tìm
searchInput.addEventListener('keyup', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

// Hàm này render danh sách nhà hàng ra HTML
function renderRestaurantList(restaurants) {
    restaurantListEl.innerHTML = restaurants.map(r => {
        const imageUrl = r.image_src 
                       ? r.image_src 
                       : 'https://placehold.co/600x400/e2e8f0/64748b?text=Không+có+ảnh';
        
        const hasGps = r.gps && r.gps.includes(',');
        const directionButton = hasGps 
            ? `<button class="action-button navigate-button w-full mt-auto bg-blue-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-600" data-gps="${r.gps}" data-name="${r.name}">Chỉ đường</button>`
            : `<button class="w-full mt-auto bg-gray-300 text-gray-500 font-bold py-2 px-4 rounded-lg cursor-not-allowed" disabled>Không có GPS</button>`;
        
        return `
        <div class="bg-white rounded-lg shadow-md overflow-hidden flex flex-col">
            <img src="${imageUrl}" alt="Ảnh quán ${r.name}" class="w-full h-48 object-cover" onerror="this.src='https://placehold.co/600x400/e2e8f0/64748b?text=Lỗi+tải+ảnh';">
            <div class="p-4 flex flex-col flex-grow">
                <h2 class="text-lg font-semibold text-gray-800 flex-grow">${r.name}</h2>
                <p class="text-sm text-gray-600 mt-1 mb-4">${r.address || 'Không có địa chỉ'}</p>
                ${directionButton}
            </div>
        </div>`;
    }).join('');
    
    addNavigateButtonListeners();
}

// Hàm này gắn listener cho các nút "Chỉ đường"
function addNavigateButtonListeners() {
    document.querySelectorAll('.navigate-button').forEach(button => {
        // Xóa listener cũ (nếu có) để tránh gắn trùng lặp
        button.removeEventListener('click', handleNavigateClick);
        // Gắn listener mới
        button.addEventListener('click', handleNavigateClick);
    });
}

// Tách hàm xử lý click ra riêng
function handleNavigateClick() {
    const gpsString = this.getAttribute('data-gps');
    const restaurantName = this.getAttribute('data-name');
    const [destLat, destLng] = gpsString.split(',').map(c => parseFloat(c.trim()));
    currentDestination = { name: restaurantName, coords: { lat: destLat, lng: destLng } };
    openMapModal();
}
// --- CÁC HÀM XỬ LÝ MODAL VÀ BẢN ĐỒ ---
function openMapModal() {
    mapModalTitle.textContent = `Chỉ đường tới: ${currentDestination.name}`;
    mapModal.classList.remove('hidden');
    setTimeout(() => mapModal.classList.remove('opacity-0'), 10);
    
    if (!map) {
        map = L.map('map');
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    }
    setTimeout(() => map.invalidateSize(), 200);

    renderTravelModeButtons();
    calculateAndDrawRoute();
}

function renderTravelModeButtons() {
    travelModeSelector.innerHTML = TRAVEL_MODES.map(mode => `
        <button class="travel-mode-btn flex items-center p-2 rounded-lg font-semibold text-gray-600 ${mode.id === currentTravelMode ? 'active' : ''}" data-mode="${mode.id}">
            ${mode.icon}
            <span class="ml-2">${mode.label}</span>
        </button>
    `).join('');

    document.querySelectorAll('.travel-mode-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            currentTravelMode = this.dataset.mode;
            renderTravelModeButtons();
            calculateAndDrawRoute();
        });
    });
}

function calculateAndDrawRoute() {
    if (routingControl) map.removeControl(routingControl);
    if (animatedMarker) map.removeLayer(animatedMarker);
    startAnimationBtn.disabled = true;
    mapLoader.style.display = 'flex';
    summaryDistance.textContent = "--";
    summaryTime.textContent = "--";

    if (!userLocation) {
        map.setView([currentDestination.coords.lat, currentDestination.coords.lng], 15);
        L.marker([currentDestination.coords.lat, currentDestination.coords.lng]).addTo(map).bindPopup(currentDestination.name).openPopup();
        mapLoader.style.display = 'none';
        return;
    }

    const selectedMode = TRAVEL_MODES.find(m => m.id === currentTravelMode);
    
    routingControl = L.Routing.control({
        waypoints: [
            L.latLng(userLocation.lat, userLocation.lng),
            L.latLng(currentDestination.coords.lat, currentDestination.coords.lng)
        ],
        router: L.Routing.osrmv1({
            serviceUrl: `https://router.project-osrm.org/route/v1`,
            profile: selectedMode.profile
        }),
        addWaypoints: false,
        createMarker: () => null,
        lineOptions: { styles: [{ color: '#0d9488', opacity: 0.8, weight: 6 }] }
    }).on('routesfound', function(e) {
        mapLoader.style.display = 'none';
        const route = e.routes[0];
        const distanceInKm = route.summary.totalDistance / 1000;
        
        // --- NÂNG CẤP: Tính toán lại thời gian ---
        const speedKmh = CUSTOM_SPEEDS_KMH[currentTravelMode];
        const timeInMinutes = (distanceInKm / speedKmh) * 60;

        summaryDistance.textContent = `${distanceInKm.toFixed(2)} km`;
        summaryTime.textContent = `${Math.round(timeInMinutes)} phút`;
        startAnimationBtn.disabled = false;

        startAnimationBtn.onclick = () => {
            if (animatedMarker) map.removeLayer(animatedMarker);
            
            // --- NÂNG CẤP: Tính toán tốc độ mô phỏng (mét/giây) ---
            const speedMs = (speedKmh * 1000) / 3600; // Chuyển từ km/h sang m/s

            animatedMarker = L.animatedMarker(route.coordinates, {
                distance: speedMs, // Số mét di chuyển mỗi giây
                interval: 1000,    // Cập nhật vị trí mỗi giây
                icon: L.divIcon({
                    html: ICONS_SVG[currentTravelMode],
                    className: 'bg-transparent border-0',
                    iconSize: [32, 32]
                })
            });
            map.addLayer(animatedMarker);
        };
    }).addTo(map);
}

function closeMapModal() {
    if (animatedMarker) {
        animatedMarker.stop();
        map.removeLayer(animatedMarker);
        animatedMarker = null;
    }
    mapModal.classList.add('opacity-0');
    setTimeout(() => mapModal.classList.add('hidden'), 300);
}

document.getElementById('closeMapModal').addEventListener('click', closeMapModal);
mapModal.addEventListener('click', (e) => (e.target === mapModal) && closeMapModal());