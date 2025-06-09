import numpy as np
import time
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple
import json
import os


class CharacterEmbedding:
    """
    Character embedding system for improved character identification and relationship tracking.
    Uses sentence transformers to create vector representations of characters and their contexts.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the character embedding system.
        
        Args:
            model_name: The SentenceTransformer model to use for embeddings
        """
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.character_embeddings = {}
        self.context_embeddings = {}
        self.relationship_graph = {}
        
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a character name or text.
        
        Args:
            text: The text to embed
            
        Returns:
            numpy array representing the text embedding
        """
        return self.model.encode(text)
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
            return 0.0
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def add_character_embedding(self, character: str, contexts: List[str]) -> None:
        """
        Add or update character embedding based on their contexts.
        
        Args:
            character: Character name
            contexts: List of context strings where the character appears
        """
        if not contexts:
            return
            
        # Create embedding from concatenated contexts
        combined_context = " ".join(contexts[:5])  # Use first 5 contexts to avoid too long text
        character_embedding = self.get_embedding(f"{character}: {combined_context}")
        
        self.character_embeddings[character] = {
            'embedding': character_embedding,
            'contexts': contexts[:10],  # Store up to 10 contexts
            'updated_at': time.time()
        }
    
    def find_similar_character(self, candidate: str, candidate_context: str, 
                             known_characters: List[str], threshold: float = 0.7) -> Optional[str]:
        """
        Find the most similar known character using embeddings.
        
        Args:
            candidate: Candidate character name
            candidate_context: Context where the candidate appears
            known_characters: List of known character names
            threshold: Similarity threshold for matching
            
        Returns:
            Most similar character name if similarity > threshold, None otherwise
        """
        if not known_characters:
            return None
            
        candidate_embedding = self.get_embedding(f"{candidate}: {candidate_context}")
        best_match = None
        best_similarity = 0.0
        
        for known_char in known_characters:
            if known_char in self.character_embeddings:
                known_embedding = self.character_embeddings[known_char]['embedding']
                similarity = self.calculate_similarity(candidate_embedding, known_embedding)
                
                if similarity > threshold and similarity > best_similarity:
                    best_match = known_char
                    best_similarity = similarity
        
        return best_match
    
    def update_context(self, character: str, context: str, line_number: int = 0) -> None:
        """
        Update character context embedding with new information.
        
        Args:
            character: Character name
            context: New context string
            line_number: Line number where context appears
        """
        if character not in self.context_embeddings:
            self.context_embeddings[character] = []
        
        context_data = {
            'embedding': self.get_embedding(context),
            'timestamp': time.time(),
            'line_number': line_number,
            'context': context[:200],  # Store first 200 chars as reference
            'context_type': self._classify_context(context)
        }
        
        self.context_embeddings[character].append(context_data)
        
        # Keep only recent contexts (last 20 entries)
        if len(self.context_embeddings[character]) > 20:
            self.context_embeddings[character] = self.context_embeddings[character][-20:]
    
    def _classify_context(self, context: str) -> str:
        """
        Classify the type of context (dialogue, action, description, etc.).
        
        Args:
            context: Context string
            
        Returns:
            Context type classification
        """
        context_lower = context.lower().strip()
        
        if '"' in context or "'" in context:
            return "dialogue"
        elif any(word in context_lower for word in ['said', 'asked', 'replied', 'shouted', 'whispered']):
            return "speech_tag"
        elif any(word in context_lower for word in ['walked', 'ran', 'stood', 'sat', 'moved']):
            return "action"
        else:
            return "description"
    
    def calculate_character_relationship_strength(self, char1: str, char2: str) -> float:
        """
        Calculate relationship strength between two characters based on context proximity.
        
        Args:
            char1: First character name
            char2: Second character name
            
        Returns:
            Relationship strength score between 0 and 1
        """
        if char1 not in self.context_embeddings or char2 not in self.context_embeddings:
            return 0.0
        
        char1_contexts = self.context_embeddings[char1]
        char2_contexts = self.context_embeddings[char2]
        
        similarities = []
        for ctx1 in char1_contexts[-10:]:  # Use last 10 contexts
            for ctx2 in char2_contexts[-10:]:
                # Check if contexts are from nearby lines
                line_distance = abs(ctx1['line_number'] - ctx2['line_number'])
                if line_distance <= 5:  # Within 5 lines of each other
                    similarity = self.calculate_similarity(ctx1['embedding'], ctx2['embedding'])
                    # Weight by proximity
                    proximity_weight = 1.0 - (line_distance / 5.0)
                    similarities.append(similarity * proximity_weight)
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_character_relationships(self, character: str, min_strength: float = 0.3) -> List[Tuple[str, float]]:
        """
        Get relationships for a character above minimum strength threshold.
        
        Args:
            character: Character name
            min_strength: Minimum relationship strength to include
            
        Returns:
            List of (other_character, strength) tuples
        """
        relationships = []
        
        for other_char in self.context_embeddings:
            if other_char != character:
                strength = self.calculate_character_relationship_strength(character, other_char)
                if strength >= min_strength:
                    relationships.append((other_char, strength))
        
        # Sort by strength descending
        relationships.sort(key=lambda x: x[1], reverse=True)
        return relationships
    
    def save_embeddings(self, filepath: str) -> None:
        """
        Save embeddings and relationship data to file.
        
        Args:
            filepath: Path to save the data
        """
        # Convert numpy arrays to lists for JSON serialization
        save_data = {
            'character_embeddings': {},
            'context_embeddings': {},
            'metadata': {
                'model_name': self.model._modules['0'].auto_model.name_or_path,
                'created_at': time.time()
            }
        }
        
        for char, data in self.character_embeddings.items():
            save_data['character_embeddings'][char] = {
                'embedding': data['embedding'].tolist(),
                'contexts': data['contexts'],
                'updated_at': data['updated_at']
            }
        
        for char, contexts in self.context_embeddings.items():
            save_data['context_embeddings'][char] = []
            for ctx in contexts:
                save_data['context_embeddings'][char].append({
                    'embedding': ctx['embedding'].tolist(),
                    'timestamp': ctx['timestamp'],
                    'line_number': ctx['line_number'],
                    'context': ctx['context'],
                    'context_type': ctx['context_type']
                })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    def load_embeddings(self, filepath: str) -> None:
        """
        Load embeddings and relationship data from file.
        
        Args:
            filepath: Path to load the data from
        """
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        # Convert lists back to numpy arrays
        for char, data in save_data.get('character_embeddings', {}).items():
            self.character_embeddings[char] = {
                'embedding': np.array(data['embedding']),
                'contexts': data['contexts'],
                'updated_at': data['updated_at']
            }
        
        for char, contexts in save_data.get('context_embeddings', {}).items():
            self.context_embeddings[char] = []
            for ctx in contexts:
                self.context_embeddings[char].append({
                    'embedding': np.array(ctx['embedding']),
                    'timestamp': ctx['timestamp'],
                    'line_number': ctx['line_number'],
                    'context': ctx['context'],
                    'context_type': ctx['context_type']
                })


class CharacterRelationshipGraph:
    """
    Graph-based character relationship tracking using embeddings.
    """
    
    def __init__(self, embedding_system: CharacterEmbedding):
        """
        Initialize relationship graph with embedding system.
        
        Args:
            embedding_system: CharacterEmbedding instance
        """
        self.embedding_system = embedding_system
        self.relationships = {}
        self.interaction_history = []
    
    def add_interaction(self, char1: str, char2: str, context: str, line_number: int) -> None:
        """
        Add an interaction between two characters.
        
        Args:
            char1: First character
            char2: Second character  
            context: Context of interaction
            line_number: Line number where interaction occurs
        """
        interaction = {
            'characters': sorted([char1, char2]),
            'context': context,
            'line_number': line_number,
            'timestamp': time.time(),
            'embedding': self.embedding_system.get_embedding(context)
        }
        
        self.interaction_history.append(interaction)
        
        # Update relationship strength
        pair = tuple(sorted([char1, char2]))
        if pair not in self.relationships:
            self.relationships[pair] = []
        
        self.relationships[pair].append(interaction)
    
    def get_relationship_strength(self, char1: str, char2: str) -> float:
        """
        Get relationship strength between two characters.
        
        Args:
            char1: First character
            char2: Second character
            
        Returns:
            Relationship strength score
        """
        pair = tuple(sorted([char1, char2]))
        if pair not in self.relationships:
            return 0.0
        
        interactions = self.relationships[pair]
        if not interactions:
            return 0.0
        
        # Calculate strength based on frequency and recency
        total_interactions = len(interactions)
        recent_interactions = len([i for i in interactions if time.time() - i['timestamp'] < 3600])  # Last hour
        
        base_strength = min(total_interactions / 10.0, 1.0)  # Cap at 1.0
        recency_bonus = recent_interactions / max(total_interactions, 1) * 0.2
        
        return base_strength + recency_bonus
    
    def get_character_network(self, character: str) -> Dict[str, float]:
        """
        Get network of relationships for a character.
        
        Args:
            character: Character name
            
        Returns:
            Dictionary mapping other characters to relationship strengths
        """
        network = {}
        
        for pair, interactions in self.relationships.items():
            if character in pair:
                other_char = pair[0] if pair[1] == character else pair[1]
                network[other_char] = self.get_relationship_strength(character, other_char)
        
        return network 