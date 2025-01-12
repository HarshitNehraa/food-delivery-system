import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from models.dqn_model import DQNAgent, generate_riders, generate_orders, FoodDeliveryEnv
import tensorflow as tf
import requests
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Configuration
class Config:
    RIDERS_COUNT = 10
    ORDERS_COUNT = 20
    MODEL_PATH = os.path.join('models', 'model.keras')
    DEBUG = True
    SECRET_KEY = 'your-secret-key-here'


def osrm_distance_time(lat1, lon1, lat2, lon2):
    """Calculate distance and time using OSRM"""
    try:
        base_url = "http://router.project-osrm.org/route/v1/driving/"
        coords = f"{lon1},{lat1};{lon2},{lat2}"
        url = f"{base_url}{coords}?overview=false"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            route = data['routes'][0]
            distance = route['distance'] / 1000  # Convert to km
            duration = route['duration'] / 3600  # Convert to hours
            return distance, duration
    except Exception as e:
        logger.error(f"OSRM API error: {str(e)}")
    return 0.0, 0.0


# Initialize global variables
try:
    riders = generate_riders(Config.RIDERS_COUNT)
    orders = generate_orders(Config.ORDERS_COUNT)
    env = FoodDeliveryEnv(riders, orders)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize agent
    agent = DQNAgent(state_size, action_size)

    # Thread lock for synchronization
    order_lock = Lock()

    logger.info("Application initialized successfully")
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    raise


@app.route('/')
def index():
    """Render main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return render_template('error.html', error=str(e)), 500


@app.route('/dashboard')
def dashboard():
    """Render dashboard page"""
    try:
        return render_template('dashboard.html',
                               riders=riders.to_dict('records'),
                               orders=orders.to_dict('records'))
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        return render_template('error.html', error=str(e)), 500


@app.route('/api/assign_order', methods=['POST'])
def assign_order():
    """Assign order to rider using DQN model"""
    try:
        with order_lock:
            data = request.json
            if not data:
                return jsonify({
                    'success': False,
                    'message': 'No data provided'
                }), 400

            # Validate input data
            required_fields = ['pickup_location', 'delivery_location', 'order_weight']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'success': False,
                        'message': f'Missing required field: {field}'
                    }), 400

            pickup = data['pickup_location']
            delivery = data['delivery_location']
            weight = float(data['order_weight'])

            # Validate coordinates
            if not all(key in pickup for key in ['lat', 'lng']) or \
                    not all(key in delivery for key in ['lat', 'lng']):
                return jsonify({
                    'success': False,
                    'message': 'Invalid location format'
                }), 400

            # Get available riders
            available_riders = riders[riders['Available']].index.tolist()
            if not available_riders:
                return jsonify({
                    'success': False,
                    'message': 'No riders available'
                }), 400

            # Get current state
            state = env.get_state()

            # Get model prediction
            action = agent.act(state)

            # Convert action to rider assignment
            rider_idx = action % len(available_riders)
            actual_rider_idx = available_riders[rider_idx]

            # Calculate distance and time
            pickup_to_delivery_dist, duration = osrm_distance_time(
                float(pickup['lat']), float(pickup['lng']),
                float(delivery['lat']), float(delivery['lng'])
            )

            # Update rider and order status
            riders.at[actual_rider_idx, 'Available'] = False
            riders.at[actual_rider_idx, 'Latitude'] = float(delivery['lat'])
            riders.at[actual_rider_idx, 'Longitude'] = float(delivery['lng'])

            # Add new order to orders DataFrame
            new_order = {
                'OrderID': len(orders) + 1,
                'PickupLatitude': float(pickup['lat']),
                'PickupLongitude': float(pickup['lng']),
                'DeliveryLatitude': float(delivery['lat']),
                'DeliveryLongitude': float(delivery['lng']),
                'OrderWeight': weight,
                'OrderTime': datetime.now(),
                'Assigned': True,
                'RiderID': riders.iloc[actual_rider_idx]['RiderID']
            }
            orders.loc[len(orders)] = new_order

            # Calculate estimated delivery time
            estimated_time = max(30, int(duration * 60))  # Minimum 30 minutes

            logger.info(f"Order assigned to Rider #{riders.iloc[actual_rider_idx]['RiderID']}")

            return jsonify({
                'success': True,
                'rider_id': int(riders.iloc[actual_rider_idx]['RiderID']),
                'estimated_time': f"{estimated_time} mins"
            })

    except Exception as e:
        logger.error(f"Error assigning order: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error assigning order: {str(e)}'
        }), 500


@app.route('/api/get_riders', methods=['GET'])
def get_riders():
    """Get available riders"""
    try:
        available_riders = riders[riders['Available']].to_dict('records')
        return jsonify(available_riders)
    except Exception as e:
        logger.error(f"Error getting riders: {str(e)}")
        return jsonify([]), 500


@app.route('/api/get_orders', methods=['GET'])
def get_orders():
    """Get active orders"""
    try:
        active_orders = orders.tail(20).to_dict('records')
        return jsonify(active_orders)
    except Exception as e:
        logger.error(f"Error getting orders: {str(e)}")
        return jsonify([]), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500


def initialize_app():
    """Initialize the application"""
    try:
        # Create necessary directories
        os.makedirs('models', exist_ok=True)

        # Initialize environment
        env.reset()
        logger.info("Environment initialized")

        # Load or initialize model
        model_path = Config.MODEL_PATH
        if os.path.exists(model_path):
            try:
                agent.load(model_path)
                logger.info("Loaded pre-trained model")
            except Exception as e:
                logger.warning(f"Could not load model: {str(e)}")
                logger.info("Using new model")
        else:
            logger.info("No pre-trained model found, using new model")

    except Exception as e:
        logger.error(f"Error initializing app: {str(e)}")
        raise


if __name__ == '__main__':
    initialize_app()
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)