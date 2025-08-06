"""
Advanced retrieval-free context compression algorithms.

This module implements state-of-the-art compression techniques including
hierarchical compression, importance sampling, and semantic filtering.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import re
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .context_compressor import ContextCompressor, CompressionMetrics, CompressionConfig
from .logging_config import get_logger
from .exceptions import ValidationError, ConfigurationError

logger = get_logger("advanced_compressor")


@dataclass
class ImportanceScore:
    """Importance score for a text segment."""
    segment_id: str
    text: str
    position_score: float
    semantic_score: float
    frequency_score: float
    length_score: float
    novelty_score: float
    final_score: float


class SemanticFilter(ContextCompressor):
    """Semantic filtering based context compressor."""
    
    async def _load_models(self) -> None:
        """Load semantic similarity models."""
        try:
            self.model = SentenceTransformer(self.config.model_name)
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            self.model = self.model.to(device)
            
            # Load TF-IDF vectorizer for keyword extraction
            self.tfidf = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.logger.info(f"Loaded semantic filter models on {device}")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise ConfigurationError(f"Could not load semantic models: {e}")
    
    async def compress(
        self, 
        text: str, 
        target_length: Optional[int] = None,
        preserve_structure: bool = True,
        query: Optional[str] = None
    ) -> Tuple[str, CompressionMetrics]:
        """
        Compress text using semantic filtering.
        
        Filters sentences based on semantic similarity to key topics
        or a provided query/context.
        """
        start_time = datetime.now()
        original_tokens = self._count_tokens(text)
        
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 3:
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
        
        target_sentences = max(3, int(len(sentences) * target_ratio))
        
        try:
            # Get sentence embeddings
            sentence_embeddings = self.model.encode(sentences, batch_size=self.config.batch_size)
            
            # If query provided, filter by similarity to query
            if query:
                query_embedding = self.model.encode([query])
                similarities = cosine_similarity(sentence_embeddings, query_embedding).flatten()
            else:
                # Extract key topics using TF-IDF
                tfidf_matrix = self.tfidf.fit_transform(sentences)
                
                # Get top TF-IDF terms as representative topics
                feature_names = self.tfidf.get_feature_names_out()
                tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
                top_terms_indices = np.argsort(tfidf_scores)[-10:]  # Top 10 terms
                top_terms = [feature_names[i] for i in top_terms_indices]
                
                # Create pseudo-query from top terms
                pseudo_query = " ".join(top_terms)
                query_embedding = self.model.encode([pseudo_query])
                similarities = cosine_similarity(sentence_embeddings, query_embedding).flatten()
            
            # Score sentences with multiple factors
            sentence_scores = []
            for i, (sentence, similarity) in enumerate(zip(sentences, similarities)):
                # Base semantic similarity
                semantic_score = similarity
                
                # Position importance
                position_score = 0.0
                if i < self.config.preserve_first_sentences:
                    position_score = 0.3
                elif i >= len(sentences) - self.config.preserve_last_sentences:
                    position_score = 0.2
                
                # Length normalization
                length_score = min(len(sentence.split()) / 20, 1.0)
                
                # Information density (unique words ratio)
                words = sentence.split()
                unique_words = set(word.lower() for word in words if word.isalpha())
                density_score = len(unique_words) / max(len(words), 1) if words else 0
                
                final_score = (
                    0.6 * semantic_score +
                    0.2 * position_score +
                    0.1 * length_score +
                    0.1 * density_score
                )
                
                sentence_scores.append((i, sentence, final_score))
            
            # Filter by semantic threshold
            filtered_scores = [
                (i, sentence, score) for i, sentence, score in sentence_scores
                if score >= self.config.semantic_threshold
            ]
            
            # If too few sentences pass threshold, lower it
            if len(filtered_scores) < target_sentences:
                sorted_scores = sorted(sentence_scores, key=lambda x: x[2], reverse=True)
                filtered_scores = sorted_scores[:target_sentences]
            else:
                # Take top scoring sentences within filtered set
                filtered_scores.sort(key=lambda x: x[2], reverse=True)
                filtered_scores = filtered_scores[:target_sentences]
            
            # Reconstruct text preserving order
            if preserve_structure:
                selected_indices = sorted([score[0] for score in filtered_scores])
                compressed_sentences = [sentences[i] for i in selected_indices]
            else:
                compressed_sentences = [score[1] for score in filtered_scores]
            
            compressed_text = " ".join(compressed_sentences)
            compressed_tokens = self._count_tokens(compressed_text)
            
        except Exception as e:
            self.logger.error(f"Semantic filtering failed: {e}")
            # Fallback to simple truncation
            target_sentences = max(3, int(len(sentences) * target_ratio))
            compressed_sentences = sentences[:target_sentences]
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
            information_retention=semantic_similarity
        )
        
        self.logger.info(
            f"Semantic filtering: {original_tokens} → {compressed_tokens} tokens "
            f"({compression_ratio:.2f} ratio, {semantic_similarity:.3f} similarity)"
        )
        
        return compressed_text, metrics


class ImportanceSampler(ContextCompressor):
    """Importance sampling based context compressor."""
    
    async def _load_models(self) -> None:
        """Load models for importance scoring."""
        try:
            self.model = SentenceTransformer(self.config.model_name)
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            self.model = self.model.to(device)
            
            # Initialize NER pipeline for entity extraction
            try:
                self.ner_pipeline = pipeline(
                    "ner", 
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=0 if device == "cuda" else -1
                )
            except Exception:
                self.ner_pipeline = None
                self.logger.warning("NER pipeline not available, using fallback scoring")
            
            self.logger.info(f"Loaded importance sampling models on {device}")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise ConfigurationError(f"Could not load importance models: {e}")
    
    def _calculate_importance_scores(
        self, 
        sentences: List[str], 
        embeddings: np.ndarray
    ) -> List[ImportanceScore]:
        """Calculate comprehensive importance scores for sentences."""
        scores = []
        
        # Calculate document statistics
        doc_length = len(sentences)
        all_words = " ".join(sentences).lower().split()
        word_freq = {}
        for word in all_words:
            if word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate sentence novelty (how different from previous content)
        novelty_window = min(5, doc_length // 4)  # Look back window
        
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            
            # 1. Position score (beginning and end are important)
            if i < 2:
                position_score = 1.0 - (i * 0.1)
            elif i >= doc_length - 2:
                position_score = 0.8
            else:
                # Middle sections get lower scores
                middle_position = abs(i - doc_length / 2) / (doc_length / 2)
                position_score = 0.3 + 0.4 * (1 - middle_position)
            
            # 2. Semantic score (similarity to document centroid)
            doc_centroid = np.mean(embeddings, axis=0)
            semantic_score = cosine_similarity([embeddings[i]], [doc_centroid])[0][0]
            semantic_score = max(0, semantic_score)  # Ensure non-negative
            
            # 3. Frequency score (inverse document frequency)
            rare_word_bonus = 0
            for word in words:
                if word.isalpha() and word in word_freq:
                    # Boost for rare words
                    if word_freq[word] <= 2:
                        rare_word_bonus += 0.1
            frequency_score = min(1.0, rare_word_bonus)
            
            # 4. Length score (moderate length preferred)
            optimal_length = 15  # words
            length_diff = abs(len(words) - optimal_length)
            length_score = max(0, 1.0 - (length_diff / optimal_length))
            
            # 5. Novelty score (how different from previous content)
            novelty_score = 1.0
            if i > 0:
                # Compare with previous sentences in window
                window_start = max(0, i - novelty_window)
                prev_embeddings = embeddings[window_start:i]
                if len(prev_embeddings) > 0:
                    prev_centroid = np.mean(prev_embeddings, axis=0)
                    similarity_to_prev = cosine_similarity(
                        [embeddings[i]], [prev_centroid]
                    )[0][0]
                    novelty_score = 1.0 - max(0, similarity_to_prev)
            
            # Extract named entities if available
            entity_bonus = 0
            if self.ner_pipeline:
                try:
                    entities = self.ner_pipeline(sentence)
                    entity_bonus = min(0.3, len(entities) * 0.1)
                except Exception:
                    pass
            
            # Combine all scores with weights
            final_score = (
                0.25 * position_score +
                0.30 * semantic_score +
                0.15 * frequency_score +
                0.15 * length_score +
                0.15 * novelty_score +
                entity_bonus
            )
            
            importance = ImportanceScore(
                segment_id=f"sent_{i}",
                text=sentence,
                position_score=position_score,
                semantic_score=semantic_score,
                frequency_score=frequency_score,
                length_score=length_score,
                novelty_score=novelty_score,
                final_score=final_score
            )
            
            scores.append(importance)
        
        return scores
    
    async def compress(
        self, 
        text: str, 
        target_length: Optional[int] = None,
        preserve_structure: bool = True
    ) -> Tuple[str, CompressionMetrics]:
        """
        Compress text using importance sampling.
        
        Uses sophisticated importance scoring to probabilistically sample
        the most important sentences while maintaining diversity.
        """
        start_time = datetime.now()
        original_tokens = self._count_tokens(text)
        
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 3:
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
        
        target_sentences = max(3, int(len(sentences) * target_ratio))
        
        try:
            # Get sentence embeddings
            sentence_embeddings = self.model.encode(sentences, batch_size=self.config.batch_size)
            
            # Calculate importance scores
            importance_scores = self._calculate_importance_scores(sentences, sentence_embeddings)
            
            # Importance sampling with guaranteed minimums
            selected_indices = set()
            
            # Always include first and last sentences if preserving structure
            if preserve_structure and len(sentences) > 2:
                selected_indices.add(0)
                selected_indices.add(len(sentences) - 1)
            
            # Sample remaining sentences based on importance scores
            remaining_slots = target_sentences - len(selected_indices)
            if remaining_slots > 0:
                # Create probability distribution based on importance scores
                scores = np.array([score.final_score for score in importance_scores])
                
                # Boost scores for unselected sentences
                available_indices = [i for i in range(len(sentences)) if i not in selected_indices]
                available_scores = scores[available_indices]
                
                # Normalize to probabilities
                if np.sum(available_scores) > 0:
                    probabilities = available_scores / np.sum(available_scores)
                    
                    # Sample without replacement
                    sampled_indices = np.random.choice(
                        available_indices,
                        size=min(remaining_slots, len(available_indices)),
                        replace=False,
                        p=probabilities
                    )
                    
                    selected_indices.update(sampled_indices)
                else:
                    # Fallback: select highest scoring sentences
                    available_scores_with_indices = [
                        (i, scores[i]) for i in available_indices
                    ]
                    available_scores_with_indices.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, _ in available_scores_with_indices[:remaining_slots]:
                        selected_indices.add(i)
            
            # Reconstruct text
            if preserve_structure:
                selected_indices = sorted(list(selected_indices))
                compressed_sentences = [sentences[i] for i in selected_indices]
            else:
                # Order by importance score
                selected_scores = [(i, importance_scores[i]) for i in selected_indices]
                selected_scores.sort(key=lambda x: x[1].final_score, reverse=True)
                compressed_sentences = [importance_scores[i].text for i, _ in selected_scores]
            
            compressed_text = " ".join(compressed_sentences)
            compressed_tokens = self._count_tokens(compressed_text)
            
        except Exception as e:
            self.logger.error(f"Importance sampling failed: {e}")
            # Fallback to random sampling with position bias
            target_sentences = max(3, int(len(sentences) * target_ratio))
            
            # Create position-biased probabilities
            position_weights = np.ones(len(sentences))
            position_weights[:2] = 2.0  # Boost first sentences
            position_weights[-2:] = 1.5  # Boost last sentences
            probabilities = position_weights / np.sum(position_weights)
            
            selected_indices = np.random.choice(
                len(sentences),
                size=target_sentences,
                replace=False,
                p=probabilities
            )
            
            if preserve_structure:
                selected_indices = sorted(selected_indices)
            
            compressed_sentences = [sentences[i] for i in selected_indices]
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
            information_retention=semantic_similarity
        )
        
        self.logger.info(
            f"Importance sampling: {original_tokens} → {compressed_tokens} tokens "
            f"({compression_ratio:.2f} ratio, {len(selected_indices)} sentences selected)"
        )
        
        return compressed_text, metrics


class HierarchicalCompressor(ContextCompressor):
    """Hierarchical compression with multi-level processing."""
    
    async def _load_models(self) -> None:
        """Load models for hierarchical compression."""
        try:
            self.model = SentenceTransformer(self.config.model_name)
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            self.model = self.model.to(device)
            
            # Initialize sub-compressors
            from .context_compressor import ExtractiveSummarizer, SentenceClusterer
            
            self.extractive_compressor = ExtractiveSummarizer(self.config)
            await self.extractive_compressor.initialize()
            
            self.cluster_compressor = SentenceClusterer(self.config)
            await self.cluster_compressor.initialize()
            
            self.logger.info(f"Loaded hierarchical compression models on {device}")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise ConfigurationError(f"Could not load hierarchical models: {e}")
    
    async def compress(
        self, 
        text: str, 
        target_length: Optional[int] = None,
        preserve_structure: bool = True
    ) -> Tuple[str, CompressionMetrics]:
        """
        Compress text using hierarchical multi-stage approach.
        
        Applies different compression strategies at different levels:
        1. Document level: Major section identification
        2. Section level: Paragraph importance
        3. Paragraph level: Sentence selection
        """
        start_time = datetime.now()
        original_tokens = self._count_tokens(text)
        
        if original_tokens < 500:  # Small documents
            # Use simple extractive summarization
            return await self.extractive_compressor.compress(text, target_length, preserve_structure)
        
        # Calculate target compression ratio
        if target_length:
            target_ratio = min(target_length / original_tokens, 1.0)
        else:
            target_ratio = self.config.target_ratio
        
        try:
            # Stage 1: Document structure analysis
            sections = self._identify_sections(text)
            
            if len(sections) <= 2:
                # No clear structure, use clustering
                return await self.cluster_compressor.compress(text, target_length, preserve_structure)
            
            # Stage 2: Section-level compression
            compressed_sections = []
            total_section_importance = 0
            section_importances = []
            
            for section in sections:
                # Calculate section importance
                section_sentences = self._split_into_sentences(section)
                section_embeddings = self.model.encode(section_sentences)
                section_centroid = np.mean(section_embeddings, axis=0)
                
                # Global document embedding
                all_sentences = self._split_into_sentences(text)
                doc_embeddings = self.model.encode(all_sentences)
                doc_centroid = np.mean(doc_embeddings, axis=0)
                
                # Importance based on similarity to document centroid
                importance = cosine_similarity([section_centroid], [doc_centroid])[0][0]
                section_importances.append(importance)
                total_section_importance += importance
            
            # Allocate compression budget proportionally
            for i, section in enumerate(sections):
                if total_section_importance > 0:
                    section_ratio = section_importances[i] / total_section_importance
                    section_target_ratio = min(1.0, target_ratio * (1 + section_ratio))
                else:
                    section_target_ratio = target_ratio
                
                section_tokens = self._count_tokens(section)
                section_target_length = int(section_tokens * section_target_ratio)
                
                # Compress section using extractive summarization
                compressed_section, _ = await self.extractive_compressor.compress(
                    section, section_target_length, preserve_structure
                )
                
                if compressed_section.strip():
                    compressed_sections.append(compressed_section)
            
            # Stage 3: Final assembly and polishing
            compressed_text = "\n\n".join(compressed_sections)
            
            # If still too long, apply final compression
            compressed_tokens = self._count_tokens(compressed_text)
            if target_length and compressed_tokens > target_length * 1.1:
                # Apply final round of compression
                final_target = int(target_length * 0.9)  # Leave some buffer
                compressed_text, _ = await self.extractive_compressor.compress(
                    compressed_text, final_target, preserve_structure
                )
            
            final_tokens = self._count_tokens(compressed_text)
            
        except Exception as e:
            self.logger.error(f"Hierarchical compression failed: {e}")
            # Fallback to extractive summarization
            return await self.extractive_compressor.compress(text, target_length, preserve_structure)
        
        # Calculate final metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        compression_ratio = final_tokens / original_tokens if original_tokens > 0 else 1.0
        semantic_similarity = self._calculate_semantic_similarity(text, compressed_text)
        
        metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=final_tokens,
            compression_ratio=compression_ratio,
            processing_time=processing_time,
            semantic_similarity=semantic_similarity,
            information_retention=semantic_similarity
        )
        
        self.logger.info(
            f"Hierarchical compression: {original_tokens} → {final_tokens} tokens "
            f"({compression_ratio:.2f} ratio, {len(sections)} sections processed)"
        )
        
        return compressed_text, metrics
    
    def _identify_sections(self, text: str) -> List[str]:
        """Identify document sections based on structural patterns."""
        # Look for common section markers
        section_patterns = [
            r'\n\n#{1,6}\s+',  # Markdown headers
            r'\n\n\d+\.\s+',   # Numbered sections
            r'\n\n[A-Z][^.]*:\s*\n',  # Title: pattern
            r'\n\n[A-Z\s]{5,}\n',     # ALL CAPS titles
            r'\n\n\*\*[^*]+\*\*\s*\n', # Bold titles
        ]
        
        # Try to split by patterns
        for pattern in section_patterns:
            sections = re.split(pattern, text)
            if len(sections) > 2:
                # Clean up sections
                cleaned_sections = []
                for section in sections:
                    section = section.strip()
                    if len(section) > 100:  # Minimum section length
                        cleaned_sections.append(section)
                
                if len(cleaned_sections) > 2:
                    return cleaned_sections
        
        # Fallback: split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 100]
        
        if len(paragraphs) <= 2:
            # No clear structure, return as single section
            return [text]
        
        # Group consecutive paragraphs into sections
        sections = []
        current_section = []
        current_length = 0
        target_section_length = len(text) // min(5, len(paragraphs))
        
        for paragraph in paragraphs:
            current_section.append(paragraph)
            current_length += len(paragraph)
            
            if current_length >= target_section_length:
                sections.append('\n\n'.join(current_section))
                current_section = []
                current_length = 0
        
        # Add remaining paragraphs
        if current_section:
            if sections:
                # Merge with last section if it's small
                if len('\n\n'.join(current_section)) < target_section_length // 2:
                    sections[-1] += '\n\n' + '\n\n'.join(current_section)
                else:
                    sections.append('\n\n'.join(current_section))
            else:
                sections.append('\n\n'.join(current_section))
        
        return sections if sections else [text]