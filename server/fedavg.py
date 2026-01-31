"""
Federated Averaging (FedAvg) Algorithm
Aggregates model weights from multiple clients
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FedAvg:
    """
    Federated Averaging algorithm implementation
    Aggregates model weights from multiple clients
    """
    
    def __init__(self, base_model_path: str):
        """
        Initialize FedAvg aggregator
        
        Args:
            base_model_path: Path to the initial global model
        """
        self.base_model_path = base_model_path
        self.global_model = None
        self.current_round = 0
        logger.info(f"FedAvg initialized with base model: {base_model_path}")
    
    def load_global_model(self):
        """Load the global model"""
        try:
            self.global_model = tf.keras.models.load_model(self.base_model_path, compile=False)
            logger.info("Global model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load global model: {e}")
            return False
    
    def aggregate_weights(
        self, 
        client_weights: List[List[np.ndarray]], 
        client_samples: List[int] = None
    ) -> List[np.ndarray]:
        """
        Aggregate weights from multiple clients using weighted averaging
        
        Args:
            client_weights: List of weight arrays from each client
            client_samples: Number of samples each client trained on (for weighting)
                          If None, uses simple average
        
        Returns:
            Aggregated weights
        """
        num_clients = len(client_weights)
        
        if num_clients == 0:
            logger.warning("No client weights to aggregate")
            return None
        
        logger.info(f"Aggregating weights from {num_clients} clients")
        
        # If no sample counts provided, use equal weights
        if client_samples is None:
            client_samples = [1] * num_clients
        
        # Normalize weights based on number of samples
        total_samples = sum(client_samples)
        weights = [n / total_samples for n in client_samples]
        
        logger.info(f"Client weights: {weights}")
        
        # Initialize aggregated weights with zeros
        aggregated_weights = []
        
        # Get number of layers
        num_layers = len(client_weights[0])
        
        # Aggregate each layer
        for layer_idx in range(num_layers):
            # Weighted sum of this layer from all clients
            layer_sum = np.zeros_like(client_weights[0][layer_idx])
            
            for client_idx, client_weight in enumerate(client_weights):
                layer_sum += weights[client_idx] * client_weight[layer_idx]
            
            aggregated_weights.append(layer_sum)
        
        logger.info(f"Aggregated {num_layers} layers")
        return aggregated_weights
    
    def aggregate_deltas(
        self,
        client_deltas: List[List[np.ndarray]],
        client_samples: List[int] = None
    ) -> List[np.ndarray]:
        """
        Aggregate weight deltas from multiple clients
        
        Args:
            client_deltas: List of weight delta arrays from each client
            client_samples: Number of samples each client trained on
        
        Returns:
            Aggregated deltas to apply to global model
        """
        logger.info(f"Aggregating deltas from {len(client_deltas)} clients")
        
        # Aggregate deltas same way as weights
        aggregated_deltas = self.aggregate_weights(client_deltas, client_samples)
        
        return aggregated_deltas
    
    def update_global_model(
        self,
        client_weights: List[List[np.ndarray]] = None,
        client_deltas: List[List[np.ndarray]] = None,
        client_samples: List[int] = None,
        use_deltas: bool = False
    ) -> bool:
        """
        Update the global model with aggregated weights
        
        Args:
            client_weights: Full weights from clients
            client_deltas: Weight deltas from clients
            client_samples: Number of samples per client
            use_deltas: Whether to use deltas or full weights
        
        Returns:
            Success status
        """
        if self.global_model is None:
            logger.error("Global model not loaded")
            return False
        
        try:
            if use_deltas and client_deltas is not None:
                # Aggregate deltas
                aggregated_deltas = self.aggregate_deltas(client_deltas, client_samples)
                
                # Apply deltas to current global model
                current_weights = self.global_model.get_weights()
                new_weights = [
                    current_w + delta 
                    for current_w, delta in zip(current_weights, aggregated_deltas)
                ]
                
                logger.info("Applied aggregated deltas to global model")
            
            elif client_weights is not None:
                # Aggregate full weights
                new_weights = self.aggregate_weights(client_weights, client_samples)
                logger.info("Aggregated full weights for global model")
            
            else:
                logger.error("No weights or deltas provided")
                return False
            
            # Update global model
            self.global_model.set_weights(new_weights)
            self.current_round += 1
            
            logger.info(f"Global model updated - Round {self.current_round}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update global model: {e}")
            return False
    
    def save_global_model(self, save_path: str) -> bool:
        """
        Save the updated global model
        
        Args:
            save_path: Path to save the model
        
        Returns:
            Success status
        """
        try:
            self.global_model.save(save_path)
            logger.info(f"Global model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save global model: {e}")
            return False
    
    def get_global_weights(self) -> List[np.ndarray]:
        """
        Get current global model weights
        
        Returns:
            Global model weights
        """
        if self.global_model is None:
            logger.error("Global model not loaded")
            return None
        
        return self.global_model.get_weights()
    
    def get_stats(self) -> Dict:
        """
        Get aggregator statistics
        
        Returns:
            Dictionary with stats
        """
        return {
            'current_round': self.current_round,
            'model_loaded': self.global_model is not None,
            'base_model_path': self.base_model_path
        }
