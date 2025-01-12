// Map Service Class
class MapService {
    constructor() {
        this.map = null;
        this.markers = [];
        this.route = null;
        this.pickupMarker = null;
        this.deliveryMarker = null;
    }

    init() {
        // Initialize Leaflet map
        this.map = L.map('map').setView([28.7041, 77.1025], 12);

        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);

        // Initialize event listeners
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const pickupInput = document.getElementById('pickupLocation');
        const deliveryInput = document.getElementById('deliveryLocation');

        if (pickupInput) {
            pickupInput.addEventListener('input', debounce((e) =>
                this.handleLocationSearch(e.target.value, 'pickup'), 500));
        }

        if (deliveryInput) {
            deliveryInput.addEventListener('input', debounce((e) =>
                this.handleLocationSearch(e.target.value, 'delivery'), 500));
        }
    }

    async handleLocationSearch(query, type) {
        if (query.length < 3) return;

        try {
            showLoading();
            const results = await this.searchLocation(query);
            this.showSearchResults(results, type);
        } catch (error) {
            showNotification('Error searching location', 'error');
        } finally {
            hideLoading();
        }
    }

    async searchLocation(query) {
        const response = await fetch(
            `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`
        );
        return await response.json();
    }

    showSearchResults(results, type) {
        const resultsDiv = document.getElementById(`${type}Results`);
        if (!resultsDiv) return;

        resultsDiv.style.display = 'block';

        resultsDiv.innerHTML = results.map(result => `
            <div class="search-result-item" 
                 onclick="mapService.selectLocation(
                     ${result.lat}, 
                     ${result.lon}, 
                     '${type}', 
                     '${result.display_name.replace(/'/g, "\\'")}'
                 )">
                ${result.display_name}
            </div>
        `).join('');
    }

    selectLocation(lat, lng, type, displayName) {
        const latInput = document.getElementById(`${type}Lat`);
        const lngInput = document.getElementById(`${type}Lng`);
        const locationInput = document.getElementById(`${type}Location`);
        const resultsDiv = document.getElementById(`${type}Results`);

        if (latInput) latInput.value = lat;
        if (lngInput) lngInput.value = lng;
        if (locationInput) locationInput.value = displayName;
        if (resultsDiv) resultsDiv.style.display = 'none';

        this.updateMapMarker(lat, lng, type);
        this.updateRoute();
    }

    updateMapMarker(lat, lng, type) {
        const markerObj = type === 'pickup' ? this.pickupMarker : this.deliveryMarker;
        const icon = L.icon({
            iconUrl: `/static/images/${type}-marker.png`,
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34]
        });

        if (markerObj) {
            this.map.removeLayer(markerObj);
        }

        const marker = L.marker([lat, lng], { icon }).addTo(this.map);

        if (type === 'pickup') {
            this.pickupMarker = marker;
        } else {
            this.deliveryMarker = marker;
        }

        this.fitMapToBounds();
    }

    fitMapToBounds() {
        const bounds = L.latLngBounds([]);

        if (this.pickupMarker) {
            bounds.extend(this.pickupMarker.getLatLng());
        }
        if (this.deliveryMarker) {
            bounds.extend(this.deliveryMarker.getLatLng());
        }

        if (bounds.isValid()) {
            this.map.fitBounds(bounds, {
                padding: [50, 50],
                maxZoom: 15
            });
        }
    }

    async updateRoute() {
        if (!this.pickupMarker || !this.deliveryMarker) return;

        const pickup = this.pickupMarker.getLatLng();
        const delivery = this.deliveryMarker.getLatLng();

        try {
            showLoading();

            const url = `https://router.project-osrm.org/route/v1/driving/`
                + `${pickup.lng},${pickup.lat};${delivery.lng},${delivery.lat}`
                + `?overview=full&geometries=geojson`;

            const response = await fetch(url);
            const data = await response.json();

            if (this.route) {
                this.map.removeLayer(this.route);
            }

            if (data.routes && data.routes.length > 0) {
                this.route = L.geoJSON(data.routes[0].geometry, {
                    style: {
                        color: '#007bff',
                        weight: 4,
                        opacity: 0.6
                    }
                }).addTo(this.map);
            }
        } catch (error) {
            showNotification('Error calculating route', 'error');
        } finally {
            hideLoading();
        }
    }

    clearMap() {
        if (this.pickupMarker) this.map.removeLayer(this.pickupMarker);
        if (this.deliveryMarker) this.map.removeLayer(this.deliveryMarker);
        if (this.route) this.map.removeLayer(this.route);

        this.pickupMarker = null;
        this.deliveryMarker = null;
        this.route = null;
    }

    async getCurrentLocation(type) {
        if (!navigator.geolocation) {
            showNotification('Geolocation is not supported by your browser', 'error');
            return;
        }

        try {
            showLoading();
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject);
            });

            const { latitude, longitude } = position.coords;
            const response = await fetch(
                `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}`
            );
            const data = await response.json();

            this.selectLocation(latitude, longitude, type, data.display_name);
        } catch (error) {
            showNotification('Error getting current location', 'error');
        } finally {
            hideLoading();
        }
    }
}

// Order Management Class
class OrderManager {
    constructor() {
        this.orders = [];
    }

    async submitOrder(orderData) {
        try {
            showLoading();

            if (!this.validateOrderData(orderData)) {
                return;
            }

            const response = await fetch('/api/assign_order', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(orderData)
            });

            const data = await response.json();

            if (data.success) {
                showNotification(`Order assigned to Rider #${data.rider_id}. 
                    Estimated delivery time: ${data.estimated_time}`, 'success');
                document.getElementById('orderForm').reset();
                mapService.clearMap();
                this.updateActiveOrders();
            } else {
                throw new Error(data.message || 'Error assigning order');
            }
        } catch (error) {
            showNotification(error.message, 'error');
        } finally {
            hideLoading();
        }
    }

    validateOrderData(orderData) {
        if (!orderData.pickup_location?.lat || !orderData.pickup_location?.lng) {
            showNotification('Please select a pickup location', 'error');
            return false;
        }

        if (!orderData.delivery_location?.lat || !orderData.delivery_location?.lng) {
            showNotification('Please select a delivery location', 'error');
            return false;
        }

        if (!orderData.order_weight || orderData.order_weight <= 0) {
            showNotification('Please enter a valid order weight', 'error');
            return false;
        }

        return true;
    }

    async updateActiveOrders() {
        try {
            const response = await fetch('/api/get_orders');
            const orders = await response.json();

            const activeOrdersDiv = document.getElementById('activeOrders');
            if (!activeOrdersDiv) return;

            activeOrdersDiv.innerHTML = orders.map(order => `
                <div class="col-md-6 col-lg-4 mb-3">
                    <div class="order-card">
                        <h5>Order #${order.OrderID}</h5>
                        <p class="mb-2">
                            <span class="status-badge ${order.Assigned ? 'status-assigned' : 'status-pending'}">
                                ${order.Assigned ? 'Assigned' : 'Pending'}
                            </span>
                        </p>
                        <p class="mb-2">Weight: ${order.OrderWeight} kg</p>
                        <button class="btn btn-sm btn-outline-primary" 
                                onclick="orderManager.showOrderOnMap(${order.PickupLatitude}, 
                                    ${order.PickupLongitude}, 
                                    ${order.DeliveryLatitude}, 
                                    ${order.DeliveryLongitude})">
                            Show on Map
                        </button>
                    </div>
                </div>
            `).join('');
        } catch (error) {
            showNotification('Error updating orders', 'error');
        }
    }

    showOrderOnMap(pickupLat, pickupLng, deliveryLat, deliveryLng) {
        mapService.clearMap();
        mapService.updateMapMarker(pickupLat, pickupLng, 'pickup');
        mapService.updateMapMarker(deliveryLat, deliveryLng, 'delivery');
        mapService.updateRoute();
    }
}

// Utility Functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function showLoading() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = 'flex';
    }
}

function hideLoading() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = 'none';
    }
}

function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Initialize services
const mapService = new MapService();
const orderManager = new OrderManager();

// Document ready
document.addEventListener('DOMContentLoaded', () => {
    mapService.init();

    // Initialize order form
    const orderForm = document.getElementById('orderForm');
    if (orderForm) {
        orderForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const orderData = {
                pickup_location: {
                    lat: document.getElementById('pickupLat').value,
                    lng: document.getElementById('pickupLng').value
                },
                delivery_location: {
                    lat: document.getElementById('deliveryLat').value,
                    lng: document.getElementById('deliveryLng').value
                },
                order_weight: document.getElementById('orderWeight').value
            };

            await orderManager.submitOrder(orderData);
        });
    }

    // Initialize filters
    const showOnlyAvailable = document.getElementById('showOnlyAvailable');
    if (showOnlyAvailable) {
        showOnlyAvailable.addEventListener('change', function() {
            const riderRows = document.querySelectorAll('.rider-row');
            riderRows.forEach(row => {
                if (this.checked) {
                    row.style.display = row.querySelector('.badge').classList.contains('bg-success') ? '' : 'none';
                } else {
                    row.style.display = '';
                }
            });
        });
    }

    // Update orders periodically
    orderManager.updateActiveOrders();
    setInterval(() => orderManager.updateActiveOrders(), 30000);
});