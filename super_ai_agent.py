"""
SUPER AI AGENT - Learning LM-Style Desktop Application
========================================================
A self-learning AI agent with:
- Neural network-inspired pattern learning
- Memory consolidation system
- Context-aware responses
- Evolution through interactions
- Visual learning feedback

Author: AI Engineer
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
import random
import re
import hashlib
import math
import pickle
import os
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import threading
import time

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Neuron:
    """Represents a learning neuron with weights and activation"""
    id: str
    weight: float = 0.5
    activation: float = 0.0
    bias: float = 0.1
    learning_rate: float = 0.1
    activation_history: List[float] = field(default_factory=list)
    
    def activate(self, input_signal: float) -> float:
        """Sigmoid activation function"""
        self.activation = 1 / (1 + math.exp(-(input_signal * self.weight + self.bias)))
        self.activation_history.append(self.activation)
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
        return self.activation
    
    def learn(self, error: float, input_signal: float):
        """Hebbian learning: neurons that fire together wire together"""
        delta = self.learning_rate * error * input_signal
        self.weight += delta
        # Clamp weights
        self.weight = max(-1, min(1, self.weight))

@dataclass
class MemoryTrace:
    """Represents a consolidated memory"""
    id: str
    content: str
    emotional_weight: float
    frequency: int = 1
    last_accessed: str = ""
    connections: List[str] = field(default_factory=list)
    importance: float = 0.5
    
    def to_dict(self):
        return asdict(self)
    
    @staticmethod
    def from_dict(data):
        return MemoryTrace(**data)

@dataclass
class ConversationContext:
    """Maintains conversation context for coherent responses"""
    topics: List[str] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)
    sentiment_trend: List[float] = field(default_factory=list)
    user_style: Dict[str, Any] = field(default_factory=dict)
    recent_responses: List[str] = field(default_factory=list)

# ============================================================================
# NEURAL LEARNING ENGINE
# ============================================================================

class NeuralLearningEngine:
    """
    Simulates LM-style learning with neural network concepts
    """
    
    def __init__(self, num_neurons: int = 50):
        self.neurons: Dict[str, Neuron] = {}
        self.pattern_weights: Dict[str, float] = {}
        self.response_associations: Dict[str, List[str]] = defaultdict(list)
        self.context_embeddings: Dict[str, List[float]] = {}
        self.learning_events: List[Dict] = []
        
        # Initialize neurons
        for i in range(num_neurons):
            neuron_id = f"neuron_{i}"
            self.neurons[neuron_id] = Neuron(id=neuron_id)
        
        # Load or initialize pattern weights
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize base pattern recognition"""
        base_patterns = [
            "greeting", "farewell", "question", "statement", "command",
            "emotion_positive", "emotion_negative", "uncertainty",
            "agreement", "disagreement", "request_help", "sharing_info"
        ]
        for pattern in base_patterns:
            self.pattern_weights[pattern] = 0.5
    
    def embed_text(self, text: str, dimensions: int = 10) -> List[float]:
        """Create simple text embedding based on character patterns"""
        embedding = []
        for i in range(dimensions):
            # Create hash-based features
            feature_text = text[i::dimensions] if i < len(text) else text
            hash_val = int(hashlib.md5(feature_text.encode()).hexdigest()[:8], 16)
            normalized = (hash_val % 1000) / 1000.0
            embedding.append(normalized)
        return embedding
    
    def recognize_pattern(self, text: str) -> Tuple[str, float]:
        """Recognize pattern in input text"""
        text_lower = text.lower()
        
        patterns = {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening", "apa kabar", "halo"],
            "farewell": ["bye", "goodbye", "see you", "sampai jumpa", "dadah", "keluar"],
            "question": ["?", "apa", "siapa", "kapan", "di mana", "bagaimana", "mengapa", "kenapa"],
            "emotion_positive": ["senang", "bahagia", "suka", "cinta", "hebat", "mantap", "keren"],
            "emotion_negative": ["sedih", "marah", "benci", "kecewa", "buruk", "jelek"],
            "uncertainty": ["mungkin", "barangkali", "entahlah", "tidak tahu"],
            "agreement": ["ya", "iya", "setuju", "benar", "ok", "oke"],
            "disagreement": ["tidak", "nggak", "salah", "beda"],
            "request_help": ["bantu", "tolong", "bisa", "ajarkan"],
            "sharing_info": ["saya", "aku", "gue", "gw", "my", "i am"]
        }
        
        best_pattern = "statement"
        best_confidence = 0.3
        
        for pattern, keywords in patterns.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            confidence = matches / max(len(keywords), 1)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_pattern = pattern
        
        # Adjust by learned weights
        adjusted_confidence = best_confidence * self.pattern_weights.get(best_pattern, 0.5)
        
        return best_pattern, adjusted_confidence
    
    def learn_from_interaction(self, input_text: str, response: str, feedback: float = 0.5):
        """Learn from interaction using Hebbian principles"""
        input_embedding = self.embed_text(input_text)
        response_embedding = self.embed_text(response)
        
        # Update pattern weights
        pattern, confidence = self.recognize_pattern(input_text)
        self.pattern_weights[pattern] += 0.01 * feedback * confidence
        
        # Associate input patterns with successful responses
        if feedback > 0.3:
            if response not in self.response_associations[pattern]:
                self.response_associations[pattern].append(response)
            # Strengthen association
            if len(self.response_associations[pattern]) > 10:
                self.response_associations[pattern].pop(0)
        
        # Update neuron activations
        for i, neuron_id in enumerate(self.neurons.keys()):
            if i < len(input_embedding):
                neuron = self.neurons[neuron_id]
                error = feedback - neuron.activation
                neuron.learn(error, input_embedding[i])
        
        # Record learning event
        self.learning_events.append({
            "timestamp": datetime.now().isoformat(),
            "pattern": pattern,
            "feedback": feedback,
            "input_length": len(input_text),
            "response_length": len(response)
        })
        
        # Keep only recent events
        if len(self.learning_events) > 1000:
            self.learning_events = self.learning_events[-1000:]
    
    def get_best_response(self, pattern: str, context: ConversationContext) -> str:
        """Generate response based on learned patterns"""
        candidates = self.response_associations.get(pattern, [])
        
        if candidates and random.random() < 0.7:
            # Use learned response
            return random.choice(candidates)
        
        # Fallback to template-based response
        return self._generate_template_response(pattern, context)
    
    def _generate_template_response(self, pattern: str, context: ConversationContext) -> str:
        """Generate template-based response with variation"""
        templates = {
            "greeting": [
                "Halo! Senang bertemu denganmu lagi.",
                "Hai! Ada yang bisa saya bantu?",
                "Hello! Bagaimana harimu?",
                "Selamat datang! Apa kabar?"
            ],
            "farewell": [
                "Sampai jumpa! Hati-hati di jalan.",
                "Bye! Jangan lupa kembali ya.",
                "Sampai nanti! Semoga harimu menyenangkan.",
                "Dadah! Take care!"
            ],
            "question": [
                "Pertanyaan menarik! Mari saya pikirkan...",
                "Hmm, itu pertanyaan yang bagus. Menurut saya...",
                "Biarkan saya analisis dulu...",
                "Pertanyaan mendalam! Ini pemikiran saya..."
            ],
            "emotion_positive": [
                "Wah, senang sekali mendengarnya!",
                "Itu hebat! Saya ikut bahagia untukmu.",
                "Mantap! Energi positifmu menular.",
                "Keren! Terus pertahankan semangat ini!"
            ],
            "emotion_negative": [
                "Saya mengerti perasaanmu. Ingin cerita lebih lanjut?",
                "Tidak apa-apa merasa seperti itu. Saya di sini untukmu.",
                "Tenang, kita bisa melalui ini bersama.",
                "Peluk virtual! Kadang hari-hari sulit memang terjadi."
            ],
            "uncertainty": [
                "Tidak masalah jika belum yakin. Kita bisa eksplorasi bersama.",
                "Kadang ketidakpastian itu wajar. Mau diskusi?",
                "Perlahan-lahan saja. Tidak perlu terburu-buru.",
                "Saya di sini untuk membantumu menemukan kejelasan."
            ],
            "agreement": [
                "Senang kita sepakat!",
                "Yes! Kita satu pikiran.",
                "Bagus! Kesamaan pemahaman itu penting.",
                "Solid! Mari lanjutkan."
            ],
            "disagreement": [
                "Pendapat yang berbeda itu sehat. Ceritakan pandanganmu.",
                "Menarik, kita punya perspektif berbeda. Aku ingin dengar alasanmu.",
                "Tidak masalah tidak setuju. Diskusi yang baik!",
                "Diversity of thought itu bagus. Ayo diskusikan."
            ],
            "request_help": [
                "Tentu! Saya akan bantu semampu saya.",
                "Dengan senang hati! Apa yang perlu dibantu?",
                "Siap membantu! Jelaskan apa yang kamu butuhkan.",
                "Aku di sini untuk membantu. Yuk, mulai!"
            ],
            "sharing_info": [
                "Terima kasih sudah berbagi! Itu menarik.",
                "Fascinating! Ceritakan lebih lanjut.",
                "Saya appreciate kamu berbagi hal ini.",
                "Ini insight yang berharga. Terima kasih!"
            ],
            "statement": [
                "Menarik! Lanjutkan.",
                "Saya mendengarkan. Ada lagi?",
                "Hmm, saya paham. Dan kemudian?",
                "Oke, saya follow. Apa selanjutnya?"
            ]
        }
        
        base_templates = templates.get(pattern, templates["statement"])
        
        # Add variation based on context
        response = random.choice(base_templates)
        
        # Occasionally reference past conversations
        if context.topics and random.random() < 0.2:
            topic = random.choice(context.topics[-3:])
            variations = [
                f"Ngomong-ngomong tentang {topic}, ada update?",
                f"Tadi kita bahas {topic}, masih relevan kan?",
                f"Kembali ke {topic} yang kamu sebutkan..."
            ]
            if random.random() < 0.5:
                response = random.choice(variations) + " " + response
        
        return response
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about learning progress"""
        avg_weights = sum(self.pattern_weights.values()) / max(len(self.pattern_weights), 1)
        total_associations = sum(len(v) for v in self.response_associations.values())
        
        neuron_activations = [n.activation for n in self.neurons.values()]
        avg_activation = sum(neuron_activations) / max(len(neuron_activations), 1)
        
        return {
            "avg_pattern_weight": round(avg_weights, 4),
            "total_associations": total_associations,
            "avg_neuron_activation": round(avg_activation, 4),
            "learning_events": len(self.learning_events),
            "patterns_learned": len([p for p, w in self.pattern_weights.items() if w > 0.6])
        }

# ============================================================================
# MEMORY CONSOLIDATION SYSTEM
# ============================================================================

class MemoryConsolidator:
    """
    Manages long-term memory with consolidation and retrieval
    """
    
    def __init__(self):
        self.short_term_memory: List[Dict] = []
        self.long_term_memories: Dict[str, MemoryTrace] = {}
        self.memory_index: Dict[str, List[str]] = defaultdict(list)
        self.consolidation_threshold = 3
        self.decay_rate = 0.01
    
    def add_to_short_term(self, content: str, emotion: float = 0.5):
        """Add new experience to short-term memory"""
        memory_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()[:12]
        
        entry = {
            "id": memory_id,
            "content": content,
            "emotion": emotion,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }
        
        self.short_term_memory.append(entry)
        
        # Keep STM limited
        if len(self.short_term_memory) > 50:
            self.short_term_memory = self.short_term_memory[-50:]
        
        # Attempt consolidation
        self._attempt_consolidation(entry)
        
        return memory_id
    
    def _attempt_consolidation(self, entry: Dict):
        """Try to consolidate short-term memory to long-term"""
        content = entry["content"]
        
        # Check if similar memory exists
        similar_found = False
        for mem_id, memory in self.long_term_memories.items():
            if self._similarity(content, memory.content) > 0.7:
                memory.frequency += 1
                memory.last_accessed = datetime.now().isoformat()
                memory.emotional_weight = (memory.emotional_weight + entry["emotion"]) / 2
                similar_found = True
                break
        
        if not similar_found:
            # Count occurrences in STM
            occurrences = sum(1 for m in self.short_term_memory 
                            if self._similarity(m["content"], content) > 0.6)
            
            if occurrences >= self.consolidation_threshold:
                # Consolidate to LTM
                memory_id = hashlib.md5(content.encode()).hexdigest()[:12]
                
                new_memory = MemoryTrace(
                    id=memory_id,
                    content=content,
                    emotional_weight=entry["emotion"],
                    frequency=occurrences,
                    last_accessed=datetime.now().isoformat(),
                    importance=min(1.0, occurrences / 10)
                )
                
                self.long_term_memories[memory_id] = new_memory
                
                # Index by keywords
                keywords = self._extract_keywords(content)
                for kw in keywords:
                    self.memory_index[kw].append(memory_id)
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        stopwords = {"yang", "dan", "atau", "tetapi", "dengan", "untuk", "dari", "pada", "di", "ke", "dari"}
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return keywords[:5]
    
    def retrieve_relevant(self, query: str, limit: int = 3) -> List[MemoryTrace]:
        """Retrieve relevant memories based on query"""
        query_keywords = self._extract_keywords(query)
        
        candidate_scores: Dict[str, float] = {}
        
        for kw in query_keywords:
            if kw in self.memory_index:
                for mem_id in self.memory_index[kw]:
                    if mem_id in self.long_term_memories:
                        mem = self.long_term_memories[mem_id]
                        score = self._similarity(query, mem.content) * mem.importance
                        score *= mem.frequency  # Boost by frequency
                        candidate_scores[mem_id] = candidate_scores.get(mem_id, 0) + score
        
        # Sort by score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for mem_id, score in sorted_candidates[:limit]:
            if mem_id in self.long_term_memories:
                mem = self.long_term_memories[mem_id]
                mem.last_accessed = datetime.now().isoformat()
                results.append(mem)
        
        return results
    
    def apply_decay(self):
        """Apply memory decay over time"""
        for mem_id, memory in self.long_term_memories.items():
            if memory.frequency == 0:
                memory.importance *= (1 - self.decay_rate)
                if memory.importance < 0.1:
                    # Remove weak memories
                    del self.long_term_memories[mem_id]
                    for kw in self.memory_index:
                        if mem_id in self.memory_index[kw]:
                            self.memory_index[kw].remove(mem_id)
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        total_ltm = len(self.long_term_memories)
        avg_importance = sum(m.importance for m in self.long_term_memories.values()) / max(total_ltm, 1)
        avg_frequency = sum(m.frequency for m in self.long_term_memories.values()) / max(total_ltm, 1)
        
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": total_ltm,
            "avg_importance": round(avg_importance, 4),
            "avg_frequency": round(avg_frequency, 2),
            "indexed_keywords": len(self.memory_index)
        }

# ============================================================================
# SUPER AI AGENT CORE
# ============================================================================

class SuperAIAgent:
    """
    Main AI Agent combining learning, memory, and response generation
    """
    
    def __init__(self, agent_name: str = "Nexus"):
        self.name = agent_name
        self.learning_engine = NeuralLearningEngine()
        self.memory = MemoryConsolidator()
        self.context = ConversationContext()
        
        self.evolution_level = 1
        self.total_interactions = 0
        self.positive_feedback_count = 0
        self.personality_traits = {
            "friendliness": 0.5,
            "curiosity": 0.5,
            "creativity": 0.5,
            "empathy": 0.5
        }
        
        self.is_thinking = False
        self.thought_process: List[str] = []
        
        # Load saved state if exists
        self.load_state()
    
    def process_input(self, user_input: str) -> Tuple[str, Dict]:
        """Process user input and generate response"""
        self.is_thinking = True
        self.thought_process = []
        
        start_time = time.time()
        
        # Step 1: Analyze input
        self.thought_process.append("📊 Menganalisis input...")
        pattern, confidence = self.learning_engine.recognize_pattern(user_input)
        sentiment = self._analyze_sentiment(user_input)
        
        # Step 2: Retrieve relevant memories
        self.thought_process.append("🔍 Mengambil memori relevan...")
        relevant_memories = self.memory.retrieve_relevant(user_input)
        
        # Step 3: Update context
        self.thought_process.append("🔄 Memperbarui konteks...")
        self._update_context(user_input, pattern, sentiment)
        
        # Step 4: Generate response
        self.thought_process.append("💭 Membangun respons...")
        base_response = self.learning_engine.get_best_response(pattern, self.context)
        
        # Enhance with memories
        enhanced_response = self._enhance_with_memories(base_response, relevant_memories)
        
        # Add personality flavor
        final_response = self._add_personality_flavor(enhanced_response)
        
        # Step 5: Learn from interaction
        self.thought_process.append("🧠 Belajar dari interaksi...")
        self.total_interactions += 1
        
        # Auto-feedback based on engagement
        feedback = self._calculate_auto_feedback(user_input, final_response)
        self.learning_engine.learn_from_interaction(user_input, final_response, feedback)
        
        # Store in memory
        self.memory.add_to_short_term(f"User: {user_input}", max(0, sentiment))
        self.memory.add_to_short_term(f"AI: {final_response}", 0.5)
        
        # Check evolution
        self._check_evolution()
        
        # Apply memory decay periodically
        if self.total_interactions % 10 == 0:
            self.memory.apply_decay()
        
        processing_time = time.time() - start_time
        
        self.is_thinking = False
        
        metadata = {
            "pattern": pattern,
            "confidence": confidence,
            "sentiment": sentiment,
            "memories_used": len(relevant_memories),
            "processing_time_ms": round(processing_time * 1000, 2),
            "evolution_level": self.evolution_level,
            "thought_process": self.thought_process.copy()
        }
        
        return final_response, metadata
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (-1 to 1)"""
        positive_words = ["senang", "bahagia", "suka", "cinta", "hebat", "mantap", "keren", "good", "great", "love", "happy"]
        negative_words = ["sedih", "marah", "benci", "kecewa", "buruk", "jelek", "bad", "hate", "sad", "angry"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def _update_context(self, input_text: str, pattern: str, sentiment: float):
        """Update conversation context"""
        # Extract topics (simple noun extraction)
        words = re.findall(r'\b\w+\b', input_text.lower())
        potential_topics = [w for w in words if len(w) > 4]
        
        for topic in potential_topics[:3]:
            if topic not in self.context.topics:
                self.context.topics.append(topic)
        
        # Limit topics
        if len(self.context.topics) > 10:
            self.context.topics = self.context.topics[-10:]
        
        # Track sentiment trend
        self.context.sentiment_trend.append(sentiment)
        if len(self.context.sentiment_trend) > 20:
            self.context.sentiment_trend = self.context.sentiment_trend[-20:]
        
        # Update user style
        if "?" in input_text:
            self.context.user_style["asks_questions"] = self.context.user_style.get("asks_questions", 0) + 1
        if len(input_text) > 100:
            self.context.user_style["detailed"] = self.context.user_style.get("detailed", 0) + 1
    
    def _enhance_with_memories(self, response: str, memories: List[MemoryTrace]) -> str:
        """Enhance response with relevant memories"""
        if not memories:
            return response
        
        memory = memories[0]  # Use most relevant
        
        # Add memory reference occasionally
        if random.random() < 0.3 and memory.frequency > 1:
            prefixes = [
                f"Ingat sebelumnya kita bahas '{memory.content[:30]}...' ",
                f"Seperti yang pernah kamu sebutkan tentang '{memory.content[:20]}...' ",
                f"Berdasarkan pembicaraan kita sebelumnya mengenai '{memory.content[:25]}...' "
            ]
            response = random.choice(prefixes) + response.lower()
        
        return response
    
    def _add_personality_flavor(self, response: str) -> str:
        """Add personality traits to response"""
        # Friendliness affects emoji usage
        if self.personality_traits["friendliness"] > 0.6:
            emojis = ["😊", "✨", "💫", "🌟", "😊"]
            if random.random() < 0.4:
                response += " " + random.choice(emojis)
        
        # Curiosity affects follow-up questions
        if self.personality_traits["curiosity"] > 0.6:
            follow_ups = [
                " Kamu bagaimana pendapatnya?",
                " Ada pengalaman serupa?",
                " Mau cerita lebih lanjut?",
                " Gimana perasaanmu tentang ini?"
            ]
            if random.random() < 0.3:
                response += random.choice(follow_ups)
        
        # Creativity affects word choice variety
        if self.personality_traits["creativity"] > 0.7:
            creative_prefixes = [
                "Wow, ",
                "Luar biasa, ",
                "Fascinating! ",
                "Ini menarik, "
            ]
            if random.random() < 0.2:
                response = random.choice(creative_prefixes) + response
        
        return response
    
    def _calculate_auto_feedback(self, input_text: str, response: str) -> float:
        """Calculate automatic feedback based on interaction quality"""
        feedback = 0.5  # Base feedback
        
        # Longer, engaged conversations get higher feedback
        if len(input_text) > 50:
            feedback += 0.1
        if len(response) > 100:
            feedback += 0.1
        
        # Questions indicate engagement
        if "?" in input_text:
            feedback += 0.1
        
        # Positive sentiment boosts feedback
        sentiment = self._analyze_sentiment(input_text)
        feedback += sentiment * 0.2
        
        return max(0, min(1, feedback))
    
    def _check_evolution(self):
        """Check and update evolution level"""
        old_level = self.evolution_level
        
        if self.total_interactions >= 100:
            self.evolution_level = 5
        elif self.total_interactions >= 50:
            self.evolution_level = 4
        elif self.total_interactions >= 25:
            self.evolution_level = 3
        elif self.total_interactions >= 10:
            self.evolution_level = 2
        
        if self.evolution_level > old_level:
            # Evolve personality
            self.personality_traits["creativity"] = min(1.0, self.personality_traits["creativity"] + 0.1)
            self.personality_traits["empathy"] = min(1.0, self.personality_traits["empathy"] + 0.1)
    
    def provide_feedback(self, feedback: float):
        """Allow user to provide explicit feedback"""
        if feedback > 0.5:
            self.positive_feedback_count += 1
            self.personality_traits["friendliness"] = min(1.0, self.personality_traits["friendliness"] + 0.02)
        else:
            self.personality_traits["friendliness"] = max(0.0, self.personality_traits["friendliness"] - 0.01)
    
    def get_agent_stats(self) -> Dict:
        """Get comprehensive agent statistics"""
        learning_stats = self.learning_engine.get_learning_stats()
        memory_stats = self.memory.get_stats()
        
        return {
            "name": self.name,
            "evolution_level": self.evolution_level,
            "total_interactions": self.total_interactions,
            "positive_feedback_ratio": round(self.positive_feedback_count / max(self.total_interactions, 1), 3),
            "personality_traits": {k: round(v, 3) for k, v in self.personality_traits.items()},
            "learning": learning_stats,
            "memory": memory_stats
        }
    
    def save_state(self, filepath: str = "agent_state.json"):
        """Save agent state to file"""
        state = {
            "name": self.name,
            "evolution_level": self.evolution_level,
            "total_interactions": self.total_interactions,
            "positive_feedback_count": self.positive_feedback_count,
            "personality_traits": self.personality_traits,
            "pattern_weights": self.learning_engine.pattern_weights,
            "response_associations": dict(self.learning_engine.response_associations),
            "long_term_memories": {k: v.to_dict() for k, v in self.memory.long_term_memories.items()},
            "context_topics": self.context.topics,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self, filepath: str = "agent_state.json"):
        """Load agent state from file"""
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.name = state.get("name", self.name)
            self.evolution_level = state.get("evolution_level", 1)
            self.total_interactions = state.get("total_interactions", 0)
            self.positive_feedback_count = state.get("positive_feedback_count", 0)
            self.personality_traits.update(state.get("personality_traits", {}))
            self.learning_engine.pattern_weights.update(state.get("pattern_weights", {}))
            
            # Load response associations
            for pattern, responses in state.get("response_associations", {}).items():
                self.learning_engine.response_associations[pattern] = responses
            
            # Load memories
            for mem_id, mem_data in state.get("long_term_memories", {}).items():
                self.memory.long_term_memories[mem_id] = MemoryTrace.from_dict(mem_data)
            
            self.context.topics = state.get("context_topics", [])
            
        except Exception as e:
            print(f"Error loading state: {e}")

# ============================================================================
# GUI APPLICATION
# ============================================================================

class SuperAgentGUI:
    """
    Modern GUI for the Super AI Agent
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Super AI Agent - Nexus")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Set dark theme colors
        self.colors = {
            "bg_dark": "#1a1a2e",
            "bg_medium": "#16213e",
            "bg_light": "#0f3460",
            "accent": "#e94560",
            "text_primary": "#ffffff",
            "text_secondary": "#a0a0a0",
            "success": "#4ecca3",
            "warning": "#ffc107",
            "info": "#17a2b8"
        }
        
        self.root.configure(bg=self.colors["bg_dark"])
        
        # Initialize agent
        self.agent = SuperAIAgent("Nexus")
        
        # Build UI
        self._setup_styles()
        self._build_ui()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Start background tasks
        self._start_background_tasks()
    
    def _setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        
        style.configure("Custom.TFrame", background=self.colors["bg_dark"])
        style.configure("Card.TFrame", background=self.colors["bg_medium"])
        
        style.configure("Title.TLabel", 
                       background=self.colors["bg_dark"],
                       foreground=self.colors["text_primary"],
                       font=("Helvetica", 18, "bold"))
        
        style.configure("Subtitle.TLabel",
                       background=self.colors["bg_dark"],
                       foreground=self.colors["text_secondary"],
                       font=("Helvetica", 11))
        
        style.configure("Accent.TButton",
                       background=self.colors["accent"],
                       foreground=self.colors["text_primary"],
                       font=("Helvetica", 10, "bold"))
        
        style.map("Accent.TButton",
                 background=[("active", "#ff6b6b")])
    
    def _build_ui(self):
        """Build the main UI"""
        # Main container
        main_frame = ttk.Frame(self.root, style="Custom.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self._build_header(main_frame)
        
        # Content area (split into chat and stats)
        content_frame = ttk.Frame(main_frame, style="Custom.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left side: Chat interface
        chat_frame = self._build_chat_interface(content_frame)
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right side: Stats and controls
        stats_frame = self._build_stats_panel(content_frame)
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Status bar
        self._build_status_bar(main_frame)
    
    def _build_header(self, parent):
        """Build header section"""
        header_frame = ttk.Frame(parent, style="Card.TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(
            header_frame,
            text="🤖 Super AI Agent - Nexus",
            style="Title.TLabel"
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Subtitle
        subtitle_label = ttk.Label(
            header_frame,
            text="Belajar & Beradaptasi Seperti Language Model",
            style="Subtitle.TLabel"
        )
        subtitle_label.pack(side=tk.LEFT, padx=10, pady=15)
        
        # Control buttons
        btn_frame = ttk.Frame(header_frame, style="Card.TFrame")
        btn_frame.pack(side=tk.RIGHT, padx=20, pady=15)
        
        save_btn = ttk.Button(
            btn_frame,
            text="💾 Save",
            command=self._save_agent,
            style="Accent.TButton"
        )
        save_btn.pack(side=tk.LEFT, padx=5)
        
        load_btn = ttk.Button(
            btn_frame,
            text="📂 Load",
            command=self._load_agent
        )
        load_btn.pack(side=tk.LEFT, padx=5)
        
        reset_btn = ttk.Button(
            btn_frame,
            text="🔄 Reset",
            command=self._reset_agent
        )
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        export_btn = ttk.Button(
            btn_frame,
            text="📤 Export",
            command=self._export_data
        )
        export_btn.pack(side=tk.LEFT, padx=5)
    
    def _build_chat_interface(self, parent):
        """Build chat interface"""
        chat_container = ttk.Frame(parent, style="Card.TFrame")
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_container,
            wrap=tk.WORD,
            bg=self.colors["bg_dark"],
            fg=self.colors["text_primary"],
            font=("Consolas", 11),
            borderwidth=0,
            highlightthickness=2,
            highlightbackground=self.colors["bg_light"],
            highlightcolor=self.colors["accent"]
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure tags for message styling
        self.chat_display.tag_configure("user", foreground="#4ecca3", justify=tk.RIGHT)
        self.chat_display.tag_configure("ai", foreground="#e94560", justify=tk.LEFT)
        self.chat_display.tag_configure("system", foreground="#ffc107", justify=tk.CENTER)
        self.chat_display.tag_configure("thought", foreground="#a0a0a0", justify=tk.LEFT, spacing1=2)
        
        # Input area
        input_frame = ttk.Frame(chat_container, style="Card.TFrame")
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.input_field = tk.Text(
            input_frame,
            height=3,
            bg=self.colors["bg_dark"],
            fg=self.colors["text_primary"],
            font=("Consolas", 11),
            borderwidth=0,
            highlightthickness=2,
            highlightbackground=self.colors["bg_light"],
            highlightcolor=self.colors["accent"]
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Bind Enter key
        self.input_field.bind("<Return>", lambda e: self._send_message())
        self.input_field.bind("<Shift-Return>", lambda e: None)  # Allow new line
        
        send_btn = ttk.Button(
            input_frame,
            text="🚀 Kirim",
            command=self._send_message,
            style="Accent.TButton"
        )
        send_btn.pack(side=tk.RIGHT)
        
        # Feedback buttons
        feedback_frame = ttk.Frame(chat_container, style="Card.TFrame")
        feedback_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(feedback_frame, text="Feedback:", 
                 bg=self.colors["bg_medium"], fg=self.colors["text_secondary"]).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(feedback_frame, text="👍 Good", 
                  command=lambda: self._give_feedback(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Button(feedback_frame, text="👎 Bad", 
                  command=lambda: self._give_feedback(0.2)).pack(side=tk.LEFT, padx=2)
        
        return chat_container
    
    def _build_stats_panel(self, parent):
        """Build statistics panel"""
        stats_container = ttk.Frame(parent, style="Card.TFrame")
        
        # Evolution Level
        evolution_frame = ttk.LabelFrame(stats_container, text="📈 Evolution Level", 
                                        bg=self.colors["bg_medium"], 
                                        fg=self.colors["text_primary"],
                                        font=("Helvetica", 12, "bold"))
        evolution_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.evolution_label = ttk.Label(
            evolution_frame,
            text="Level 1",
            background=self.colors["bg_medium"],
            foreground=self.colors["accent"],
            font=("Helvetica", 24, "bold")
        )
        self.evolution_label.pack(pady=10)
        
        self.evolution_bar = ttk.Progressbar(
            evolution_frame,
            orient=tk.HORIZONTAL,
            length=200,
            mode='determinate'
        )
        self.evolution_bar.pack(pady=5)
        
        # Personality Traits
        personality_frame = ttk.LabelFrame(stats_container, text="🎭 Personality Traits",
                                          bg=self.colors["bg_medium"],
                                          fg=self.colors["text_primary"],
                                          font=("Helvetica", 12, "bold"))
        personality_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.trait_labels = {}
        traits = ["Friendliness", "Curiosity", "Creativity", "Empathy"]
        
        for trait in traits:
            frame = ttk.Frame(personality_frame, style="Card.TFrame")
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"{trait}:", 
                     bg=self.colors["bg_medium"], 
                     fg=self.colors["text_secondary"],
                     width=12, anchor=tk.W).pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=120, mode='determinate')
            bar.pack(side=tk.LEFT, padx=5)
            
            value_label = ttk.Label(frame, text="0.50", 
                                   bg=self.colors["bg_medium"], 
                                   fg=self.colors["text_primary"],
                                   width=5)
            value_label.pack(side=tk.LEFT)
            
            self.trait_labels[trait.lower()] = {"bar": bar, "label": value_label}
        
        # Learning Stats
        learning_frame = ttk.LabelFrame(stats_container, text="🧠 Learning Stats",
                                       bg=self.colors["bg_medium"],
                                       fg=self.colors["text_primary"],
                                       font=("Helvetica", 12, "bold"))
        learning_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.learning_stats_labels = {}
        stats = ["Interactions", "Patterns", "Associations", "Memories"]
        
        for stat in stats:
            frame = ttk.Frame(learning_frame, style="Card.TFrame")
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"{stat}:", 
                     bg=self.colors["bg_medium"], 
                     fg=self.colors["text_secondary"],
                     width=12, anchor=tk.W).pack(side=tk.LEFT)
            
            label = ttk.Label(frame, text="0", 
                             bg=self.colors["bg_medium"], 
                             fg=self.colors["text_primary"])
            label.pack(side=tk.LEFT)
            
            self.learning_stats_labels[stat.lower()] = label
        
        # Thought Process
        thought_frame = ttk.LabelFrame(stats_container, text="💭 Thought Process",
                                      bg=self.colors["bg_medium"],
                                      fg=self.colors["text_primary"],
                                      font=("Helvetica", 12, "bold"))
        thought_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.thought_display = scrolledtext.ScrolledText(
            thought_frame,
            height=8,
            bg=self.colors["bg_dark"],
            fg=self.colors["text_secondary"],
            font=("Consolas", 9),
            borderwidth=0
        )
        self.thought_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Refresh button
        refresh_btn = ttk.Button(
            stats_container,
            text="🔄 Refresh Stats",
            command=self._update_stats
        )
        refresh_btn.pack(pady=10)
        
        return stats_container
    
    def _build_status_bar(self, parent):
        """Build status bar"""
        status_frame = ttk.Frame(parent, style="Card.TFrame")
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(
            status_frame,
            text="✅ Ready | Mode: Learning",
            background=self.colors["bg_medium"],
            foreground=self.colors["success"],
            font=("Helvetica", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.memory_label = ttk.Label(
            status_frame,
            text="Memory: 0 STM | 0 LTM",
            background=self.colors["bg_medium"],
            foreground=self.colors["text_secondary"],
            font=("Helvetica", 10)
        )
        self.memory_label.pack(side=tk.RIGHT, padx=20, pady=10)
    
    def _send_message(self):
        """Send user message to agent"""
        user_input = self.input_field.get("1.0", tk.END).strip()
        
        if not user_input:
            return
        
        # Clear input
        self.input_field.delete("1.0", tk.END)
        
        # Display user message
        self.chat_display.insert(tk.END, f"\nYou: {user_input}\n", "user")
        self.chat_display.see(tk.END)
        
        # Update status
        self.status_label.configure(text="⏳ Thinking...", foreground=self.colors["warning"])
        self.root.update()
        
        # Process in thread to avoid freezing
        def process():
            response, metadata = self.agent.process_input(user_input)
            self.root.after(0, lambda: self._display_response(response, metadata))
        
        thread = threading.Thread(target=process)
        thread.start()
    
    def _display_response(self, response: str, metadata: Dict):
        """Display AI response"""
        # Show thought process
        self.thought_display.delete("1.0", tk.END)
        for i, thought in enumerate(metadata.get("thought_process", [])):
            self.thought_display.insert(tk.END, f"{thought}\n", "thought")
        
        # Display AI response with typing effect
        self.chat_display.insert(tk.END, f"\n{self.agent.name}: ", "ai")
        
        def type_text(text, index=0):
            if index < len(text):
                self.chat_display.insert(tk.END, text[index])
                self.chat_display.see(tk.END)
                self.root.after(30, lambda: type_text(text, index + 1))
            else:
                self.status_label.configure(text="✅ Ready | Mode: Learning", 
                                           foreground=self.colors["success"])
                self._update_stats()
        
        type_text(response)
    
    def _give_feedback(self, feedback: float):
        """Give feedback to agent"""
        self.agent.provide_feedback(feedback)
        
        if feedback > 0.5:
            self.chat_display.insert(tk.END, f"\n[System: 👍 Positive feedback recorded]\n", "system")
        else:
            self.chat_display.insert(tk.END, f"\n[System: 👎 Negative feedback recorded]\n", "system")
        
        self._update_stats()
    
    def _update_stats(self):
        """Update statistics display"""
        stats = self.agent.get_agent_stats()
        
        # Evolution level
        self.evolution_label.configure(text=f"Level {stats['evolution_level']}")
        progress = (stats['evolution_level'] - 1) / 4 * 100
        self.evolution_bar['value'] = progress
        
        # Personality traits
        trait_map = {
            "friendliness": "Friendliness",
            "curiosity": "Curiosity",
            "creativity": "Creativity",
            "empathy": "Empathy"
        }
        
        for trait_key, display_name in trait_map.items():
            value = stats['personality_traits'].get(trait_key, 0.5)
            self.trait_labels[trait_key]["bar"]["value"] = value * 100
            self.trait_labels[trait_key]["label"].configure(text=f"{value:.2f}")
        
        # Learning stats
        self.learning_stats_labels["interactions"].configure(text=str(stats['total_interactions']))
        self.learning_stats_labels["patterns"].configure(text=str(stats['learning']['patterns_learned']))
        self.learning_stats_labels["associations"].configure(text=str(stats['learning']['total_associations']))
        self.learning_stats_labels["memories"].configure(text=str(stats['memory']['long_term_count']))
        
        # Memory status
        self.memory_label.configure(
            text=f"Memory: {stats['memory']['short_term_count']} STM | {stats['memory']['long_term_count']} LTM"
        )
    
    def _save_agent(self):
        """Save agent state"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="agent_state.json"
        )
        
        if filepath:
            self.agent.save_state(filepath)
            messagebox.showinfo("Success", "Agent state saved successfully!")
            self.status_label.configure(text="✅ Saved", foreground=self.colors["success"])
    
    def _load_agent(self):
        """Load agent state"""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.agent.load_state(filepath)
                messagebox.showinfo("Success", "Agent state loaded successfully!")
                self._update_stats()
                self.status_label.configure(text="✅ Loaded", foreground=self.colors["success"])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {str(e)}")
    
    def _reset_agent(self):
        """Reset agent"""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset the agent? All learning will be lost."):
            self.agent = SuperAIAgent("Nexus")
            self.chat_display.delete("1.0", tk.END)
            self.thought_display.delete("1.0", tk.END)
            self.chat_display.insert(tk.END, "[System: Agent reset successfully]\n", "system")
            self._update_stats()
            self.status_label.configure(text="🔄 Reset Complete", foreground=self.colors["info"])
    
    def _export_data(self):
        """Export agent data as JSON"""
        stats = self.agent.get_agent_stats()
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="agent_export.json"
        )
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Success", "Data exported successfully!")
    
    def _start_background_tasks(self):
        """Start background update tasks"""
        def periodic_update():
            self._update_stats()
            self.root.after(5000, periodic_update)  # Update every 5 seconds
        
        periodic_update()
    
    def _on_closing(self):
        """Handle window closing"""
        if messagebox.askyesno("Exit", "Save agent state before exiting?"):
            self.agent.save_state()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        # Welcome message
        self.chat_display.insert(tk.END, "=" * 60 + "\n", "system")
        self.chat_display.insert(tk.END, f"Welcome to {self.agent.name} - Your Super AI Agent!\n", "system")
        self.chat_display.insert(tk.END, "=" * 60 + "\n\n", "system")
        self.chat_display.insert(tk.END, "I'm here to learn and adapt to your conversation style.\n", "ai")
        self.chat_display.insert(tk.END, "Feel free to chat, ask questions, or share your thoughts!\n\n", "ai")
        self.chat_display.insert(tk.END, "Tip: Use 👍/👎 buttons to give feedback and help me learn better.\n", "system")
        self.chat_display.insert(tk.END, "=" * 60 + "\n\n", "system")
        
        self._update_stats()
        self.root.mainloop()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    print("🚀 Starting Super AI Agent...")
    print("💡 Loading neural learning engine...")
    print("🧠 Initializing memory consolidator...")
    print("🎨 Building GUI interface...")
    
    app = SuperAgentGUI()
    app.run()

if __name__ == "__main__":
    main()
