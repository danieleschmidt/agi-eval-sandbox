"""
Retrieval-free context compression algorithms.

This module implements various context compression techniques that can reduce 
token usage while preserving semantic meaning without relying on external 
retrieval systems.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import re
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from datetime import datetime

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .logging_config import get_logger
from .exceptions import ValidationError, ConfigurationError

logger = get_logger("context_compressor")


class CompressionStrategy(Enum):
    """Available compression strategies."""
    EXTRACTIVE_SUMMARIZATION = "extractive_summarization"
    SENTENCE_CLUSTERING = "sentence_clustering" 
    SEMANTIC_FILTERING = "semantic_filtering"
    TOKEN_PRUNING = "token_pruning"
    IMPORTANCE_SAMPLING = "importance_sampling"
    HIERARCHICAL_COMPRESSION = "hierarchical_compression"


@dataclass
class CompressionMetrics:
    """Metrics for compression performance."""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    processing_time: float
    semantic_similarity: Optional[float] = None
    information_retention: Optional[float] = None


@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE_SUMMARIZATION
    target_ratio: float = 0.5  # Target compression ratio (0.1 = 90% reduction)
    min_sentence_length: int = 10
    max_sentence_length: int = 500
    preserve_first_sentences: int = 2
    preserve_last_sentences: int = 1
    semantic_threshold: float = 0.7
    importance_threshold: float = 0.5
    cluster_ratio: float = 0.3
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 32


class ContextCompressor(ABC):
    """Abstract base class for context compression algorithms."""
    
    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.tokenizer = None
        self.model = None
        self.logger = get_logger(f"compressor.{self.__class__.__name__.lower()}")
    
    async def initialize(self) -> None:
        """Initialize the compressor with required models."""
        await self._load_models()
    
    @abstractmethod
    async def _load_models(self) -> None:
        """Load required models for compression."""
        pass
    
    @abstractmethod
    async def compress(
        self, 
        text: str, 
        target_length: Optional[int] = None,
        preserve_structure: bool = True
    ) -> Tuple[str, CompressionMetrics]:
        """
        Compress the input text.
        
        Args:
            text: Input text to compress
            target_length: Target compressed length in tokens
            preserve_structure: Whether to preserve document structure
            
        Returns:
            Tuple of (compressed_text, metrics)
        """
        pass
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        if not self.tokenizer:
            # Rough estimate: 1 token ≈ 4 characters for English
            return len(text) // 4
        return len(self.tokenizer.encode(text))
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n\n)|(?<=\.)\s+(?=\d+\.)|(?<=[.!?])\s*\n'
        sentences = re.split(sentence_pattern, text.strip())
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) >= self.config.min_sentence_length and 
                len(sentence) <= self.config.max_sentence_length):
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_semantic_similarity(
        self, 
        original: str, 
        compressed: str
    ) -> float:
        """Calculate semantic similarity between original and compressed text."""
        if not self.model:
            return 0.8  # Default similarity score
        
        try:
            # Get embeddings
            original_embedding = self.model.encode([original])
            compressed_embedding = self.model.encode([compressed])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(original_embedding, compressed_embedding)[0][0]
            return float(similarity)
        except Exception as e:
            self.logger.warning(f"Failed to calculate semantic similarity: {e}")
            return 0.8


class ExtractiveSummarizer(ContextCompressor):
    """Extractive summarization based context compressor."""
    
    async def _load_models(self) -> None:
        """Load sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.config.model_name)
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            self.model = self.model.to(device)
            self.logger.info(f"Loaded SentenceTransformer model on {device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ConfigurationError(f"Could not load model {self.config.model_name}: {e}")
    
    async def compress(
        self, 
        text: str, 
        target_length: Optional[int] = None,
        preserve_structure: bool = True
    ) -> Tuple[str, CompressionMetrics]:
        """
        Compress text using extractive summarization.
        
        Selects the most important sentences based on:
        1. Position (first/last sentences have higher importance)
        2. Similarity to document centroid
        3. Length and information density
        """
        start_time = datetime.now()
        original_tokens = self._count_tokens(text)
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 3:
            # Text too short to compress meaningfully
            metrics = CompressionMetrics(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            return text, metrics
        
        # Calculate target number of sentences
        if target_length:
            target_ratio = min(target_length / original_tokens, 1.0)
        else:
            target_ratio = self.config.target_ratio
        
        target_sentences = max(
            3,  # Minimum sentences
            int(len(sentences) * target_ratio)
        )
        
        # Get sentence embeddings
        try:
            sentence_embeddings = self.model.encode(sentences, batch_size=self.config.batch_size)
        except Exception as e:
            self.logger.error(f"Failed to encode sentences: {e}")
            raise ValidationError(f"Failed to process text: {e}")
        
        # Calculate document centroid
        document_centroid = np.mean(sentence_embeddings, axis=0)
        
        # Score sentences
        sentence_scores = []
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            # Base similarity to document centroid
            similarity = cosine_similarity([embedding], [document_centroid])[0][0]
            
            # Position bonus (first and last sentences are important)
            position_bonus = 0.0
            if i < self.config.preserve_first_sentences:
                position_bonus = 0.3 * (self.config.preserve_first_sentences - i) / self.config.preserve_first_sentences
            elif i >= len(sentences) - self.config.preserve_last_sentences:
                position_bonus = 0.2
            
            # Length bonus (moderate length sentences are preferred)
            optimal_length = 100
            length_score = 1.0 - abs(len(sentence) - optimal_length) / optimal_length
            length_bonus = 0.1 * max(0, length_score)
            
            # Information density (based on unique words)
            words = set(re.findall(r'\w+', sentence.lower()))
            density_bonus = 0.1 * min(len(words) / 20, 1.0)
            
            total_score = similarity + position_bonus + length_bonus + density_bonus
            sentence_scores.append((i, sentence, total_score))
        
        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        selected_indices = sorted([
            score[0] for score in sentence_scores[:target_sentences]
        ])
        
        # Reconstruct compressed text preserving order
        if preserve_structure:
            compressed_sentences = [sentences[i] for i in selected_indices]
        else:
            compressed_sentences = [score[1] for score in sentence_scores[:target_sentences]]
        
        compressed_text = " ".join(compressed_sentences)
        compressed_tokens = self._count_tokens(compressed_text)
        
        # Calculate metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        semantic_similarity = self._calculate_semantic_similarity(text, compressed_text)
        
        metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            processing_time=processing_time,
            semantic_similarity=semantic_similarity,
            information_retention=semantic_similarity  # Approximate
        )
        
        self.logger.info(
            f"Compressed {original_tokens} → {compressed_tokens} tokens "
            f"({compression_ratio:.2f} ratio, {semantic_similarity:.3f} similarity)"
        )
        
        return compressed_text, metrics


class SentenceClusterer(ContextCompressor):
    """Sentence clustering based context compressor."""
    
    async def _load_models(self) -> None:
        """Load sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.config.model_name)
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            self.model = self.model.to(device)
            self.logger.info(f"Loaded SentenceTransformer model on {device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ConfigurationError(f"Could not load model {self.config.model_name}: {e}")
    
    async def compress(
        self, 
        text: str, 
        target_length: Optional[int] = None,
        preserve_structure: bool = True
    ) -> Tuple[str, CompressionMetrics]:
        """
        Compress text using sentence clustering.
        
        Groups similar sentences into clusters and selects representatives
        from each cluster based on centrality and importance.
        """
        start_time = datetime.now()
        original_tokens = self._count_tokens(text)
        
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 5:
            metrics = CompressionMetrics(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            return text, metrics
        
        # Calculate target clusters
        if target_length:
            target_ratio = min(target_length / original_tokens, 1.0)
        else:
            target_ratio = self.config.cluster_ratio
        
        n_clusters = max(3, int(len(sentences) * target_ratio))
        n_clusters = min(n_clusters, len(sentences) // 2)
        
        try:
            # Get sentence embeddings
            sentence_embeddings = self.model.encode(sentences, batch_size=self.config.batch_size)
            
            # Perform clustering
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=42, 
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(sentence_embeddings)
            
            # Select representative sentences from each cluster
            selected_sentences = []
            selected_indices = []
            
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_embeddings = sentence_embeddings[cluster_indices]
                cluster_center = kmeans.cluster_centers_[cluster_id]
                
                # Find sentence closest to cluster center
                distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
                representative_idx = cluster_indices[np.argmin(distances)]
                
                selected_sentences.append(sentences[representative_idx])
                selected_indices.append(representative_idx)
            
            # Sort by original order if preserving structure
            if preserve_structure:
                sorted_pairs = sorted(zip(selected_indices, selected_sentences))
                selected_sentences = [sentence for _, sentence in sorted_pairs]
            
            compressed_text = " ".join(selected_sentences)
            compressed_tokens = self._count_tokens(compressed_text)
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            # Fallback to simple selection
            target_sentences = max(3, int(len(sentences) * target_ratio))
            step = max(1, len(sentences) // target_sentences)
            selected_sentences = sentences[::step][:target_sentences]
            compressed_text = " ".join(selected_sentences)
            compressed_tokens = self._count_tokens(compressed_text)
        
        # Calculate metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        semantic_similarity = self._calculate_semantic_similarity(text, compressed_text)
        
        metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            processing_time=processing_time,
            semantic_similarity=semantic_similarity,
            information_retention=semantic_similarity
        )
        
        self.logger.info(
            f"Clustered {len(sentences)} sentences into {n_clusters} clusters, "
            f"compressed {original_tokens} → {compressed_tokens} tokens"
        )
        
        return compressed_text, metrics


class TokenPruner(ContextCompressor):
    """Token-level pruning based context compressor."""
    
    async def _load_models(self) -> None:
        """Load tokenizer for token-level processing."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.logger.info("Loaded BERT tokenizer for token pruning")
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer: {e}, using basic tokenization")
    
    async def compress(
        self, 
        text: str, 
        target_length: Optional[int] = None,
        preserve_structure: bool = True
    ) -> Tuple[str, CompressionMetrics]:
        """
        Compress text using token-level pruning.
        
        Removes less important tokens based on:
        1. Stop words and common tokens
        2. Repetitive content
        3. Low-information density regions
        """
        start_time = datetime.now()
        original_tokens = self._count_tokens(text)
        
        if target_length:
            target_ratio = min(target_length / original_tokens, 1.0)
        else:
            target_ratio = self.config.target_ratio
        
        # Define stop words and low-importance patterns
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'must'
        }
        
        # Tokenize and score tokens
        words = re.findall(r'\b\w+\b|\W+', text)
        word_scores = []
        word_freq = {}
        
        # Count word frequencies
        for word in words:
            if word.isalpha():
                word_lower = word.lower()
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Score each token
        for i, word in enumerate(words):
            if not word.strip():  # Skip whitespace
                word_scores.append((i, word, 1.0))  # Keep whitespace
                continue
            
            if word.isalpha():
                word_lower = word.lower()
                
                # Base score (inverse frequency - rare words are more important)
                freq_score = 1.0 / math.log(word_freq[word_lower] + 1)
                
                # Stop word penalty
                stop_penalty = 0.3 if word_lower in stop_words else 1.0
                
                # Length bonus (longer words often more informative)
                length_bonus = min(len(word) / 10, 1.0)
                
                # Position bonus (beginning and end are important)
                position_bonus = 1.0
                if i < len(words) * 0.1 or i > len(words) * 0.9:
                    position_bonus = 1.2
                
                final_score = freq_score * stop_penalty * length_bonus * position_bonus
                word_scores.append((i, word, final_score))
            else:
                # Keep punctuation and special characters
                word_scores.append((i, word, 0.8))
        
        # Calculate target number of tokens to keep
        target_tokens = max(
            int(len(words) * 0.3),  # Keep at least 30%
            int(len(words) * target_ratio)
        )
        
        # Sort by score and select top tokens
        word_scores.sort(key=lambda x: x[2], reverse=True)
        
        if preserve_structure:
            # Keep tokens in original order
            selected_indices = sorted([score[0] for score in word_scores[:target_tokens]])
            compressed_words = [words[i] for i in selected_indices]
        else:
            # Order by importance
            compressed_words = [score[1] for score in word_scores[:target_tokens]]
        
        compressed_text = ''.join(compressed_words)
        
        # Clean up extra whitespace
        compressed_text = re.sub(r'\s+', ' ', compressed_text).strip()
        
        compressed_tokens = self._count_tokens(compressed_text)
        
        # Calculate metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        semantic_similarity = self._calculate_semantic_similarity(text, compressed_text)
        
        metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            processing_time=processing_time,
            semantic_similarity=semantic_similarity,
            information_retention=semantic_similarity
        )
        
        self.logger.info(
            f"Token pruning: {original_tokens} → {compressed_tokens} tokens "
            f"({compression_ratio:.2f} ratio)"
        )
        
        return compressed_text, metrics


class ContextCompressionEngine:
    """Main engine for context compression with multiple strategies."""
    
    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.compressors: Dict[CompressionStrategy, ContextCompressor] = {}
        self.logger = get_logger("compression_engine")
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all compression strategies."""
        if self._initialized:
            return
        
        self.logger.info("Initializing context compression engine...")
        
        # Initialize compressors based on configuration
        compressor_classes = {
            CompressionStrategy.EXTRACTIVE_SUMMARIZATION: ExtractiveSummarizer,
            CompressionStrategy.SENTENCE_CLUSTERING: SentenceClusterer,
            CompressionStrategy.TOKEN_PRUNING: TokenPruner,
        }
        
        for strategy, compressor_class in compressor_classes.items():
            try:
                compressor = compressor_class(self.config)
                await compressor.initialize()
                self.compressors[strategy] = compressor
                self.logger.info(f"Initialized {strategy.value} compressor")
            except Exception as e:
                self.logger.error(f"Failed to initialize {strategy.value}: {e}")
        
        if not self.compressors:
            raise ConfigurationError("No compression strategies could be initialized")
        
        self._initialized = True
        self.logger.info(f"Compression engine ready with {len(self.compressors)} strategies")
    
    async def compress(
        self, 
        text: str,
        strategy: Optional[CompressionStrategy] = None,
        target_length: Optional[int] = None,
        preserve_structure: bool = True
    ) -> Tuple[str, CompressionMetrics]:
        """
        Compress text using specified or default strategy.
        
        Args:
            text: Input text to compress
            strategy: Compression strategy to use
            target_length: Target length in tokens
            preserve_structure: Whether to preserve document structure
            
        Returns:
            Tuple of (compressed_text, metrics)
        """
        if not self._initialized:
            await self.initialize()
        
        if not text or not text.strip():
            raise ValidationError("Input text cannot be empty")
        
        strategy = strategy or self.config.strategy
        
        if strategy not in self.compressors:
            available = list(self.compressors.keys())
            raise ValidationError(
                f"Strategy {strategy.value} not available. Available: {[s.value for s in available]}"
            )
        
        compressor = self.compressors[strategy]
        
        try:
            self.logger.info(f"Compressing with {strategy.value} strategy")
            return await compressor.compress(text, target_length, preserve_structure)
        except Exception as e:
            self.logger.error(f"Compression failed with {strategy.value}: {e}")
            raise
    
    async def compress_with_fallback(
        self, 
        text: str,
        strategies: Optional[List[CompressionStrategy]] = None,
        target_length: Optional[int] = None,
        preserve_structure: bool = True
    ) -> Tuple[str, CompressionMetrics]:
        """
        Compress text with fallback strategies if primary fails.
        
        Args:
            text: Input text to compress
            strategies: List of strategies to try in order
            target_length: Target length in tokens
            preserve_structure: Whether to preserve document structure
            
        Returns:
            Tuple of (compressed_text, metrics)
        """
        if not self._initialized:
            await self.initialize()
        
        if not strategies:
            strategies = [
                CompressionStrategy.EXTRACTIVE_SUMMARIZATION,
                CompressionStrategy.SENTENCE_CLUSTERING,
                CompressionStrategy.TOKEN_PRUNING
            ]
        
        last_error = None
        
        for strategy in strategies:
            if strategy not in self.compressors:
                continue
            
            try:
                return await self.compress(text, strategy, target_length, preserve_structure)
            except Exception as e:
                self.logger.warning(f"Strategy {strategy.value} failed: {e}")
                last_error = e
                continue
        
        if last_error:
            raise ValidationError(f"All compression strategies failed. Last error: {last_error}")
        else:
            raise ConfigurationError("No valid compression strategies available")
    
    def get_available_strategies(self) -> List[CompressionStrategy]:
        """Get list of available compression strategies."""
        return list(self.compressors.keys())
    
    async def benchmark_strategies(
        self, 
        text: str,
        target_length: Optional[int] = None
    ) -> Dict[CompressionStrategy, Tuple[str, CompressionMetrics]]:
        """
        Benchmark all available strategies on the same text.
        
        Args:
            text: Input text to compress
            target_length: Target length for all strategies
            
        Returns:
            Dictionary mapping strategies to their results
        """
        if not self._initialized:
            await self.initialize()
        
        results = {}
        
        for strategy in self.compressors.keys():
            try:
                result = await self.compress(text, strategy, target_length)
                results[strategy] = result
            except Exception as e:
                self.logger.error(f"Benchmark failed for {strategy.value}: {e}")
                # Add failed result
                metrics = CompressionMetrics(
                    original_tokens=len(text) // 4,
                    compressed_tokens=0,
                    compression_ratio=0.0,
                    processing_time=0.0
                )
                results[strategy] = ("", metrics)
        
        return results