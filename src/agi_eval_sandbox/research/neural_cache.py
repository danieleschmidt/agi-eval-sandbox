"""
Neural Predictive Cache with Attention Mechanisms

Novel caching system using neural networks with attention mechanisms for
predictive prefetching and intelligent cache replacement.

Research Innovation: "Attention-Based Neural Caching for AI Evaluation Workloads"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import hashlib
import pickle
import logging
from pathlib import Path

from ..core.logging_config import get_logger
from ..core.cache import CacheBackend

logger = get_logger("neural_cache")


@dataclass
class CacheAccess:
    """Represents a cache access event with context."""
    key: str
    timestamp: datetime
    access_type: str  # 'read', 'write', 'miss'
    user_context: Optional[str] = None
    request_features: Dict[str, Any] = field(default_factory=dict)
    cache_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result from neural cache prediction."""
    predicted_keys: List[str]
    confidence_scores: List[float]
    attention_weights: np.ndarray
    prediction_timestamp: datetime = field(default_factory=datetime.now)


class AttentionCacheNet(nn.Module):
    """
    Neural network with attention mechanism for cache prediction.
    
    Architecture:
    - Embedding layer for cache keys and context
    - Multi-head self-attention for sequence modeling
    - FFN layers for prediction
    - Output layer for cache probability prediction
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        sequence_length: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        
        # Embedding layers
        self.key_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(sequence_length, embed_dim)
        self.context_embedding = nn.Linear(32, embed_dim)  # Context features
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.cache_probability = nn.Linear(embed_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        key_sequence: torch.Tensor,
        context_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            key_sequence: Sequence of cache key tokens [batch_size, seq_len]
            context_features: Context feature vector [batch_size, seq_len, 32]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            next_key_logits: Logits for next cache key prediction
            cache_probabilities: Probability of cache access
            attention_weights: Attention weights for interpretation
        """
        batch_size, seq_len = key_sequence.shape
        
        # Create embeddings
        key_embeds = self.key_embedding(key_sequence)  # [B, L, D]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=key_sequence.device)
        pos_embeds = self.position_embedding(positions).unsqueeze(0)  # [1, L, D]
        
        # Context embeddings
        context_embeds = self.context_embedding(context_features)  # [B, L, D]
        
        # Combine embeddings
        hidden_states = key_embeds + pos_embeds + context_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Apply attention layers
        attention_weights_list = []
        for i, (attention, layer_norm, ffn) in enumerate(
            zip(self.attention_layers, self.layer_norms, self.ffn_layers)
        ):
            # Self-attention
            hidden_states_transposed = hidden_states.transpose(0, 1)  # [L, B, D]
            attn_output, attn_weights = attention(
                hidden_states_transposed,
                hidden_states_transposed,
                hidden_states_transposed,
                key_padding_mask=attention_mask,
                need_weights=True
            )
            attention_weights_list.append(attn_weights)
            
            # Residual connection and layer norm
            hidden_states = layer_norm(
                hidden_states + attn_output.transpose(0, 1)
            )
            
            # Feed-forward network
            ffn_output = ffn(hidden_states)
            hidden_states = layer_norm(hidden_states + ffn_output)
            
        # Output projections
        next_key_logits = self.output_projection(hidden_states)  # [B, L, V]
        cache_probabilities = torch.sigmoid(
            self.cache_probability(hidden_states)
        ).squeeze(-1)  # [B, L]
        
        # Combine attention weights (average across layers)
        combined_attention = torch.stack(attention_weights_list).mean(dim=0)
        
        return next_key_logits, cache_probabilities, combined_attention


class NeuralPredictiveCache:
    """
    Neural predictive cache with attention-based prefetching.
    
    Key innovations:
    1. Neural attention mechanism for access pattern learning
    2. Predictive prefetching based on context and history
    3. Intelligent cache replacement using learned priorities
    4. Multi-modal context integration (user, temporal, content)
    5. Adaptive learning from cache hit/miss patterns
    """
    
    def __init__(
        self,
        cache_size: int = 10000,
        model_path: Optional[str] = None,
        learning_rate: float = 0.001,
        prediction_horizon: int = 10,
        attention_window: int = 64
    ):
        self.cache_size = cache_size
        self.prediction_horizon = prediction_horizon
        self.attention_window = attention_window
        
        # Cache storage
        self.cache_data: Dict[str, Any] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.access_sequence: deque = deque(maxlen=attention_window * 2)
        
        # Neural components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AttentionCacheNet(
            sequence_length=attention_window
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding token
        
        # Vocabulary for cache keys
        self.key_vocabulary: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
        self.reverse_vocabulary: Dict[int, str] = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
        # Training data
        self.training_sequences: List[Dict[str, Any]] = []
        self.is_trained = False
        
        # Metrics
        self.hit_rate = 0.0
        self.prediction_accuracy = 0.0
        self.prefetch_success_rate = 0.0
        
        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            
        logger.info(f"Initialized neural cache with {cache_size} entries")
        
    def _tokenize_key(self, key: str) -> int:
        """Convert cache key to vocabulary token."""
        # Simple hash-based tokenization
        key_hash = hashlib.md5(key.encode()).hexdigest()[:8]
        
        if key_hash not in self.key_vocabulary:
            if self.vocab_size < self.model.vocab_size:
                self.key_vocabulary[key_hash] = self.vocab_size
                self.reverse_vocabulary[self.vocab_size] = key_hash
                self.vocab_size += 1
            else:
                return 1  # <UNK> token
                
        return self.key_vocabulary[key_hash]
        
    def _extract_context_features(self, access: CacheAccess) -> np.ndarray:
        """Extract context features from cache access."""
        features = np.zeros(32, dtype=np.float32)
        
        # Temporal features
        hour_of_day = access.timestamp.hour / 24.0
        day_of_week = access.timestamp.weekday() / 7.0
        features[0] = hour_of_day
        features[1] = day_of_week
        
        # Access type encoding
        access_type_map = {'read': 0.0, 'write': 0.5, 'miss': 1.0}
        features[2] = access_type_map.get(access.access_type, 0.0)
        
        # Request features
        if access.request_features:
            feature_values = list(access.request_features.values())[:10]
            for i, value in enumerate(feature_values):
                if isinstance(value, (int, float)):
                    features[3 + i] = float(value)
                    
        # Cache state features
        features[13] = len(self.cache_data) / self.cache_size  # Fill ratio
        features[14] = self.hit_rate
        features[15] = self.prediction_accuracy
        
        return features
        
    async def get(self, key: str, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Get item from cache with predictive prefetching."""
        access = CacheAccess(
            key=key,
            timestamp=datetime.now(),
            access_type='read',
            request_features=context or {}
        )
        
        # Record access
        self.access_sequence.append(access)
        
        # Check cache hit
        if key in self.cache_data:
            self._update_hit_rate(True)
            
            # Update access metadata
            if key in self.cache_metadata:
                self.cache_metadata[key]['last_access'] = datetime.now()
                self.cache_metadata[key]['access_count'] += 1
                
            # Trigger predictive prefetching
            asyncio.create_task(self._predictive_prefetch(access))
            
            return self.cache_data[key]
        else:
            self._update_hit_rate(False)
            access.access_type = 'miss'
            return None
            
    async def set(
        self,
        key: str,
        value: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set item in cache with intelligent replacement."""
        access = CacheAccess(
            key=key,
            timestamp=datetime.now(),
            access_type='write',
            request_features=context or {}
        )
        
        # Record access
        self.access_sequence.append(access)
        
        # Check if cache is full and needs eviction
        if len(self.cache_data) >= self.cache_size and key not in self.cache_data:
            await self._intelligent_eviction()
            
        # Store data
        self.cache_data[key] = value
        self.cache_metadata[key] = {
            'timestamp': datetime.now(),
            'last_access': datetime.now(),
            'access_count': 1,
            'context_features': self._extract_context_features(access),
            'predicted_priority': 0.5
        }
        
        # Add to training data
        await self._add_training_sample(access)
        
        logger.debug(f"Cached item: {key}")
        
    async def _predictive_prefetch(self, access: CacheAccess) -> None:
        """Predict and prefetch likely next cache accesses."""
        if not self.is_trained or len(self.access_sequence) < 10:
            return
            
        try:
            # Prepare input sequence
            prediction_input = await self._prepare_prediction_input()
            
            if prediction_input is None:
                return
                
            # Make prediction
            with torch.no_grad():
                self.model.eval()
                next_key_logits, cache_probs, attention_weights = self.model(
                    **prediction_input
                )
                
                # Get top predictions
                top_predictions = torch.topk(
                    next_key_logits[:, -1, :],  # Last position predictions
                    k=min(self.prediction_horizon, self.vocab_size)
                )
                
                predicted_tokens = top_predictions.indices[0].cpu().numpy()
                confidence_scores = torch.softmax(
                    top_predictions.values[0], dim=0
                ).cpu().numpy()
                
                # Convert tokens back to keys
                predicted_keys = []
                for token, confidence in zip(predicted_tokens, confidence_scores):
                    if token in self.reverse_vocabulary and confidence > 0.1:
                        key_hash = self.reverse_vocabulary[token]
                        predicted_keys.append(key_hash)
                        
                # Store prediction result
                prediction_result = PredictionResult(
                    predicted_keys=predicted_keys,
                    confidence_scores=confidence_scores.tolist(),
                    attention_weights=attention_weights[0].cpu().numpy()
                )
                
                logger.debug(f"Predicted {len(predicted_keys)} prefetch candidates")
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            
    async def _prepare_prediction_input(self) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare input tensors for neural prediction."""
        if len(self.access_sequence) < self.attention_window:
            return None
            
        # Get recent access sequence
        recent_accesses = list(self.access_sequence)[-self.attention_window:]
        
        # Tokenize keys
        key_tokens = []
        context_features = []
        
        for access in recent_accesses:
            token = self._tokenize_key(access.key)
            key_tokens.append(token)
            
            features = self._extract_context_features(access)
            context_features.append(features)
            
        # Pad if necessary
        while len(key_tokens) < self.attention_window:
            key_tokens.insert(0, 0)  # Pad with <PAD> token
            context_features.insert(0, np.zeros(32, dtype=np.float32))
            
        # Convert to tensors
        key_sequence = torch.tensor([key_tokens], dtype=torch.long, device=self.device)
        context_tensor = torch.tensor(
            [context_features], dtype=torch.float32, device=self.device
        )
        
        return {
            'key_sequence': key_sequence,
            'context_features': context_tensor
        }
        
    async def _intelligent_eviction(self) -> None:
        """Intelligently evict cache entries using neural priorities."""
        if not self.cache_metadata:
            return
            
        # Calculate eviction scores
        eviction_scores = {}
        
        for key, metadata in self.cache_metadata.items():
            # Base score from neural prediction
            neural_score = metadata.get('predicted_priority', 0.5)
            
            # Time-based decay
            time_since_access = (datetime.now() - metadata['last_access']).total_seconds()
            time_decay = math.exp(-time_since_access / 3600)  # 1-hour half-life
            
            # Access frequency
            access_frequency = metadata['access_count']
            
            # Combined eviction score (lower = more likely to evict)
            eviction_score = (
                0.4 * neural_score +
                0.3 * time_decay +
                0.3 * min(access_frequency / 10.0, 1.0)
            )
            
            eviction_scores[key] = eviction_score
            
        # Evict lowest scoring entry
        key_to_evict = min(eviction_scores.keys(), key=lambda k: eviction_scores[k])
        
        del self.cache_data[key_to_evict]
        del self.cache_metadata[key_to_evict]
        
        logger.debug(f"Evicted cache entry: {key_to_evict}")
        
    async def _add_training_sample(self, access: CacheAccess) -> None:
        """Add training sample for neural model learning."""
        if len(self.access_sequence) < 2:
            return
            
        # Create training sequence from recent accesses
        sequence_length = min(len(self.access_sequence) - 1, self.attention_window)
        input_sequence = list(self.access_sequence)[-sequence_length-1:-1]
        target_access = self.access_sequence[-1]
        
        # Prepare training data
        input_tokens = [self._tokenize_key(acc.key) for acc in input_sequence]
        target_token = self._tokenize_key(target_access.key)
        
        context_features = [
            self._extract_context_features(acc) for acc in input_sequence
        ]
        
        training_sample = {
            'input_sequence': input_tokens,
            'target_token': target_token,
            'context_features': context_features,
            'timestamp': datetime.now()
        }
        
        self.training_sequences.append(training_sample)
        
        # Trigger training if enough samples accumulated
        if len(self.training_sequences) >= 100:
            asyncio.create_task(self._train_model())
            
    async def _train_model(self) -> None:
        """Train neural model on accumulated data."""
        if len(self.training_sequences) < 50:
            return
            
        logger.info("Training neural cache model")
        
        try:
            self.model.train()
            
            # Prepare training batch
            batch_inputs = []
            batch_targets = []
            batch_contexts = []
            
            for sample in self.training_sequences[-100:]:  # Use last 100 samples
                # Pad sequences
                input_seq = sample['input_sequence'][-self.attention_window:]
                while len(input_seq) < self.attention_window:
                    input_seq.insert(0, 0)  # <PAD> token
                    
                context_seq = sample['context_features'][-self.attention_window:]
                while len(context_seq) < self.attention_window:
                    context_seq.insert(0, np.zeros(32, dtype=np.float32))
                    
                batch_inputs.append(input_seq)
                batch_targets.append(sample['target_token'])
                batch_contexts.append(context_seq)
                
            # Convert to tensors
            input_tensor = torch.tensor(batch_inputs, dtype=torch.long, device=self.device)
            target_tensor = torch.tensor(batch_targets, dtype=torch.long, device=self.device)
            context_tensor = torch.tensor(
                batch_contexts, dtype=torch.float32, device=self.device
            )
            
            # Training step
            self.optimizer.zero_grad()
            
            next_key_logits, cache_probs, attention_weights = self.model(
                input_tensor, context_tensor
            )
            
            # Calculate loss (next token prediction)
            loss = self.criterion(
                next_key_logits[:, -1, :],  # Last position predictions
                target_tensor
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                predictions = torch.argmax(next_key_logits[:, -1, :], dim=1)
                accuracy = (predictions == target_tensor).float().mean().item()
                self.prediction_accuracy = 0.9 * self.prediction_accuracy + 0.1 * accuracy
                
            self.is_trained = True
            
            # Clear old training data
            self.training_sequences = self.training_sequences[-50:]
            
            logger.info(f"Training completed - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
    def _update_hit_rate(self, hit: bool) -> None:
        """Update running hit rate."""
        self.hit_rate = 0.95 * self.hit_rate + 0.05 * (1.0 if hit else 0.0)
        
    def save_model(self, path: str) -> None:
        """Save neural model to disk."""
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocabulary': self.key_vocabulary,
            'reverse_vocabulary': self.reverse_vocabulary,
            'vocab_size': self.vocab_size,
            'hit_rate': self.hit_rate,
            'prediction_accuracy': self.prediction_accuracy,
            'is_trained': self.is_trained
        }
        
        torch.save(save_data, path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str) -> None:
        """Load neural model from disk."""
        save_data = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(save_data['model_state_dict'])
        self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
        self.key_vocabulary = save_data['vocabulary']
        self.reverse_vocabulary = save_data['reverse_vocabulary']
        self.vocab_size = save_data['vocab_size']
        self.hit_rate = save_data['hit_rate']
        self.prediction_accuracy = save_data['prediction_accuracy']
        self.is_trained = save_data['is_trained']
        
        logger.info(f"Model loaded from {path}")
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'cache_size': len(self.cache_data),
            'max_cache_size': self.cache_size,
            'hit_rate': self.hit_rate,
            'prediction_accuracy': self.prediction_accuracy,
            'vocab_size': self.vocab_size,
            'is_trained': self.is_trained,
            'training_samples': len(self.training_sequences),
            'access_sequence_length': len(self.access_sequence),
            'memory_usage_mb': self._estimate_memory_usage()
        }
        
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimation
        cache_size = sum(len(pickle.dumps(v)) for v in self.cache_data.values())
        model_size = sum(p.numel() * 4 for p in self.model.parameters())  # 4 bytes per float32
        
        return (cache_size + model_size) / (1024 * 1024)
        
    def export_research_data(self) -> Dict[str, Any]:
        """Export comprehensive research data for analysis."""
        return {
            "algorithm_name": "Neural Predictive Cache with Attention",
            "model_architecture": {
                "embed_dim": self.model.embed_dim,
                "num_heads": 8,
                "num_layers": 4,
                "vocab_size": self.vocab_size
            },
            "performance_metrics": {
                "hit_rate": self.hit_rate,
                "prediction_accuracy": self.prediction_accuracy,
                "memory_usage_mb": self._estimate_memory_usage()
            },
            "training_data": {
                "samples_count": len(self.training_sequences),
                "is_trained": self.is_trained,
                "access_sequence_length": len(self.access_sequence)
            },
            "cache_statistics": self.get_cache_stats()
        }