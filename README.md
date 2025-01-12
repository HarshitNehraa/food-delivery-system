# Food Delivery System

A real-time food delivery management system built with Python Flask and modern web technologies. This system helps manage delivery riders, orders, and provides real-time tracking capabilities.

## Features

- Real-time order tracking
- Intelligent rider assignment using DQN (Deep Q-Network)
- Interactive map interface
- Order management dashboard
- Rider status monitoring
- Route optimization
- Location-based services

## Tech Stack

### Backend
- Python 3.9+
- Flask
- TensorFlow
- SQLAlchemy
- OpenStreetMap API
- OSRM (Open Source Routing Machine)

### Frontend
- HTML5
- CSS3
- JavaScript
- Leaflet.js for maps
- Bootstrap 5

## Project Structure
food_delivery_app/
│
├── static/
│ ├── css/
│ │ └── style.css
│ ├── js/
│ │ └── main.js
│ └── images/
│ ├── pickup-marker.png
│ └── delivery-marker.png
│
├── templates/
│ ├── index.html
│ └── dashboard.html
│
├── models/
│ └── dqn_model.py
│
├── app.py
└── requirements.txt



## Installation

1. Clone the repository:

git clone https://github.com/HarshitNehraa/food-delivery-system.git
cd food-delivery-system

# Create and activate virtual environment:

# Windows
python -m venv venv
venv\Scripts\activate

# Unix/MacOS
python -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Run the application:
python app.py

Environment Variables

Create a .env file in the root directory with the following variables:
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key

# Usage
Access the application at http://localhost:5000
Use the dashboard to monitor orders and riders
Place new orders through the main interface
Track deliveries in real-time on the map
API Endpoints

# Orders
POST /api/assign_order - Create and assign new order
GET /api/get_orders - Get all active orders

# Riders
GET /api/get_riders - Get available riders
POST /api/reset_rider/<rider_id> - Reset rider status

# Machine Learning Model
The system uses a Deep Q-Network (DQN) for optimal rider assignment:
State space: Rider and order locations
Action space: Rider-order assignments
Reward: Based on delivery time and distance

# Contributing
Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

# Acknowledgments
OpenStreetMap for mapping data
OSRM for routing services
TensorFlow team for ML framework
Flask team for the web framework

# Contact
Harshitnehra66@gmail.com
Project Link: https://github.com/HarshitNehraa/food-delivery-system

# Screenshots
![image](https://github.com/user-attachments/assets/1352971d-7586-4e60-9a38-0f2089c34bee)

![image](https://github.com/user-attachments/assets/67212bf1-a472-4364-9e6b-b3dfc0fe9492)

# Future Enhancements
 Mobile application
 Advanced route optimization
 Real-time chat system
 Payment integration
 Analytics dashboard
 Multi-language support
 
# Troubleshooting
Common issues and their solutions:
# Map not loading
Check internet connection
Verify API keys
# Order assignment failing
Ensure riders are available
Check database connection
# Route calculation error
Verify OSRM service status
Check coordinate validity
# Performance
Handles up to 1000 concurrent orders
Average response time: <100ms
Real-time updates every 30 seconds
# Security
JWT authentication
HTTPS encryption
Input validation
XSS protection
CSRF protection
# Backup and Recovery
Automatic database backups
State recovery system
Error logging and monitoring

# Support
For support, email harshitnehra66@gmail.com
