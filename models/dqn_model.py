import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from gym import spaces
import random
from collections import deque
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_riders(num_riders, seed=42):
    """
    Generate random rider data
    """
    np.random.seed(seed)
    return pd.DataFrame({
        'RiderID': range(1, num_riders + 1),
        'Capacity': np.random.randint(5, 15, size=num_riders),
        'MaxWeightCapacity': np.random.uniform(40, 60, size=num_riders),
        'Speed': np.random.uniform(10, 25, size=num_riders),
        'Latitude': np.random.uniform(28.5, 28.9, size=num_riders),
        'Longitude': np.random.uniform(77.0, 77.5, size=num_riders),
        'Available': np.ones(num_riders, dtype=bool)
    })


def generate_orders(num_orders, seed=43):
    """
    Generate random order data
    """
    np.random.seed(seed)
    return pd.DataFrame({
        'OrderID': range(1, num_orders + 1),
        'PickupLatitude': np.random.uniform(28.5, 28.9, size=num_orders),
        'PickupLongitude': np.random.uniform(77.0, 77.5, size=num_orders),
        'DeliveryLatitude': np.random.uniform(28.5, 28.9, size=num_orders),
        'DeliveryLongitude': np.random.uniform(77.0, 77.5, size=num_orders),
        'OrderWeight': np.random.uniform(1, 5, size=num_orders),
        'OrderTime': pd.date_range(start='2024-01-01 08:00', periods=num_orders, freq='min'),
        'Assigned': np.zeros(num_orders, dtype=bool)
    })


class FoodDeliveryEnv:
    """Food Delivery Environment"""

    def __init__(self, riders, orders):
        """Initialize environment"""
        self.riders = riders.copy()
        self.orders = orders.copy()
        self.num_riders = len(riders)
        self.num_orders = len(orders)

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_riders * 3 + self.num_orders * 3,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_riders * self.num_orders)

    def reset(self):
        """Reset environment state"""
        self.riders['Available'] = True
        self.orders['Assigned'] = False
        return self.get_state()

    def get_state(self):
        """Get current state"""
        try:
            rider_features = self.riders[['Latitude', 'Longitude', 'MaxWeightCapacity']].values.flatten()
            order_features = self.orders[['PickupLatitude', 'PickupLongitude', 'OrderWeight']].values.flatten()
            state = np.concatenate([rider_features, order_features]).astype(np.float32)
            return state
        except Exception as e:
            logger.error(f"Error getting state: {str(e)}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def step(self, action):
        """Take action in environment"""
        rider_idx = action // self.num_orders
        order_idx = action % self.num_orders

        # Check if assignment is valid
        if not self.riders.iloc[rider_idx]['Available'] or self.orders.iloc[order_idx]['Assigned']:
            return self.get_state(), -100, True, {}

        # Calculate distance-based reward
        pickup_lat = self.orders.iloc[order_idx]['PickupLatitude']
        pickup_lng = self.orders.iloc[order_idx]['PickupLongitude']
        delivery_lat = self.orders.iloc[order_idx]['DeliveryLatitude']
        delivery_lng = self.orders.iloc[order_idx]['DeliveryLongitude']
        rider_lat = self.riders.iloc[rider_idx]['Latitude']
        rider_lng = self.riders.iloc[rider_idx]['Longitude']

        # Calculate distances
        distance_to_pickup = np.sqrt((rider_lat - pickup_lat) ** 2 + (rider_lng - pickup_lng) ** 2)
        distance_delivery = np.sqrt((pickup_lat - delivery_lat) ** 2 + (pickup_lng - delivery_lng) ** 2)
        total_distance = distance_to_pickup + distance_delivery

        # Update state
        self.riders.at[rider_idx, 'Available'] = False
        self.orders.at[order_idx, 'Assigned'] = True

        # Calculate reward
        reward = 100 - total_distance * 10  # Base reward minus distance penalty

        # Check if episode is done
        done = not self.riders['Available'].any() or self.orders['Assigned'].all()

        return self.get_state(), reward, done, {}


class DQNAgent:
    """DQN Agent for learning optimal assignment policy"""

    def __init__(self, state_size, action_size):
        """Initialize DQN Agent"""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        # Create main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_network()

        logger.info("DQN Agent initialized")

    def _build_model(self):
        """Build neural network model"""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model

    def _update_target_network(self):
        """Update target network weights"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.array(state).reshape(1, -1)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Prepare batch data
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Calculate target Q values
        target_qs = rewards + self.gamma * np.amax(
            self.target_model.predict(next_states, verbose=0), axis=1
        ) * (1 - dones)

        # Get current Q values and update with targets
        current_qs = self.model.predict(states, verbose=0)
        current_qs[np.arange(len(actions)), actions] = target_qs

        # Train the model
        self.model.fit(states, current_qs, epochs=1, verbose=0)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Periodically update target network
        if random.random() < 0.1:  # 10% chance to update target network
            self._update_target_network()

    def load(self, filepath):
        """Load model weights"""
        try:
            self.model.load_weights(filepath)
            self._update_target_network()
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.warning(f"Could not load model from {filepath}: {str(e)}")
            logger.info("Using new model")

    def save(self, filepath):
        """Save model weights"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save_weights(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Could not save model to {filepath}: {str(e)}")


if __name__ == "__main__":
    # Test environment and agent
    riders = generate_riders(5)
    orders = generate_orders(10)
    env = FoodDeliveryEnv(riders, orders)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    logger.info("Test initialization successful")