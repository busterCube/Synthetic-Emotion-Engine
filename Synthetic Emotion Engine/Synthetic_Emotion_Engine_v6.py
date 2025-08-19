import sys
import os
import json
import re
import requests
import time
import sqlite_vec
import numpy as np
import threading
import warnings
from sentence_transformers import SentenceTransformer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QTextEdit, QPushButton, QSlider, QComboBox, QFileDialog,
                            QGroupBox, QGridLayout, QSplitter, QMessageBox, QProgressBar,
                            QLineEdit, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                            QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QTextCursor, QColor, QIcon

# Kokoro TTS imports
try:
    import pyaudio
    from kokoro import KPipeline
    from queue import Queue
    KOKORO_AVAILABLE = True
    print("Kokoro TTS modules loaded successfully")
except ImportError as e:
    KOKORO_AVAILABLE = False
    print(f"Kokoro TTS not available: {e}")

# Suppress Kokoro warnings
warnings.filterwarnings('ignore', message='dropout option adds dropout after all but last recurrent layer.*')
warnings.filterwarnings('ignore', message='.*torch.nn.utils.weight_norm.*is deprecated.*')

# Helper function to get paths relative to the application directory
def get_app_path(filename):
    """Get the full path to a file in the application directory"""
    if getattr(sys, 'frozen', False):
        # If running from a compiled executable
        app_dir = os.path.dirname(sys.executable)
    else:
        # If running from Python script
        app_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(app_dir, filename)

class KokoroTTS:
    """Kokoro Text-to-Speech integration for the Synthetic Emotion Engine"""
    
    def __init__(self):
        self.pipeline = None
        self.is_available = KOKORO_AVAILABLE
        self.is_speaking = False
        
        # TTS Configuration
        self.repo_id = "hexgrad/Kokoro-82M"
        self.lang_code = 'a'
        self.speed = 1.0
        self.sample_rate = 24000
        self.frames_per_buffer = 1024
        
        # Voice models
        self.fem_voices = ['af_heart','af_alloy','af_aoede','af_bella','af_jessica',
                          'af_kore','af_nicole','af_nova','af_river','af_sarah','af_sky']
        self.mal_voices = ['am_adam','am_echo','am_eric','am_fenrir','am_liam',
                          'am_michael','am_onyx','am_puck','am_santa']
        
        if self.is_available:
            try:
                print("Initializing Kokoro TTS pipeline...")
                self.pipeline = KPipeline(lang_code=self.lang_code, repo_id=self.repo_id)
                print("Kokoro TTS pipeline initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Kokoro TTS pipeline: {e}")
                self.is_available = False
    
    def split_into_sentences(self, text):
        """Split text into sentences for processing"""
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        return [p.strip() for p in parts if p.strip()]
    
    def group_sentences(self, sentences, block_size=2):
        """Group sentences into blocks for processing"""
        for i in range(0, len(sentences), block_size):
            yield ' '.join(sentences[i:i + block_size])
    
    def speak(self, text, voice=None, base_speed=None, fast_speed=None, block_size=2, initial_buffer_blocks=1):
        """Speak the given text using Kokoro TTS"""
        if not self.is_available or not self.pipeline:
            print("Kokoro TTS is not available")
            return False
        
        if self.is_speaking:
            print("TTS is already speaking")
            return False
        
        # Use default voice if none provided
        if voice is None:
            voice = self.mal_voices[0]  # Default to first male voice
        
        # Use default speeds if none provided
        if base_speed is None:
            base_speed = self.speed
        if fast_speed is None:
            fast_speed = self.speed + 0.3
        
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            print("No sentences to process for TTS")
            return False
        
        self.is_speaking = True
        print(f"Starting TTS with voice: {voice}")
        
        audio_queue = Queue(maxsize=initial_buffer_blocks + 2)
        done_sentinel = object()
        
        def producer():
            try:
                for idx, block in enumerate(self.group_sentences(sentences, block_size)):
                    speed = fast_speed if idx == 0 else base_speed
                    print(f'[TTS] Block {idx+1}: "{block[:50]}..." @ speed {speed}')
                    chunk_list = []
                    
                    for _, _, audio in self.pipeline(block, voice=voice, speed=speed):
                        chunk_list.append(np.array(audio.tolist(), dtype=np.float32))
                    
                    if chunk_list:
                        full_block = np.concatenate(chunk_list)
                        audio_queue.put(full_block)
                
                audio_queue.put(done_sentinel)
            except Exception as e:
                print(f"TTS Producer error: {e}")
                audio_queue.put(done_sentinel)
        
        def consumer():
            p = None
            stream = None
            try:
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=self.sample_rate,
                    output=True,
                    frames_per_buffer=self.frames_per_buffer
                )
                
                print(f"[TTS] Buffering {initial_buffer_blocks} block(s)...")
                for _ in range(initial_buffer_blocks):
                    blk = audio_queue.get()
                    if blk is done_sentinel:
                        return
                    stream.write(blk.tobytes())
                
                while True:
                    blk = audio_queue.get()
                    if blk is done_sentinel:
                        break
                    stream.write(blk.tobytes())
                
            except Exception as e:
                print(f"TTS Consumer error: {e}")
            finally:
                if stream:
                    stream.stop_stream()
                    stream.close()
                if p:
                    p.terminate()
                self.is_speaking = False
                print("[TTS] Playback completed")
        
        # Start producer and consumer threads
        prod_t = threading.Thread(target=producer, daemon=True)
        cons_t = threading.Thread(target=consumer, daemon=True)
        
        prod_t.start()
        cons_t.start()
        
        # Join threads in a separate daemon thread to avoid blocking UI
        def wait_for_completion():
            prod_t.join()
            cons_t.join()
        
        wait_thread = threading.Thread(target=wait_for_completion, daemon=True)
        wait_thread.start()
        
        return True
    
    def stop_speaking(self):
        """Stop current TTS playback (placeholder for future implementation)"""
        # Note: Stopping mid-playback would require more complex implementation
        # For now, we just set the flag
        self.is_speaking = False

class EmotionalVectorDB:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = get_app_path("EmotionalVectorDB.db")
        self.db_path = db_path
        self.embedding_model = None
        print("Initializing EmotionalVectorDB...")
        
        try:
            print("Loading SentenceTransformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer model loaded successfully")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            self.embedding_model = None
            
        self.setup_db()
        
    def setup_db(self):
        try:
            # Create database connection using sqlite_vec's sqlite3 module
            conn = sqlite_vec.sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            cursor = conn.cursor()
            
            # Create conversations table with emotional data
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotional_conversations (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                user_character_name TEXT,
                ai_character_name TEXT,
                user_message TEXT,
                ai_response TEXT,
                fear_level REAL,
                happiness_level REAL,
                sadness_level REAL,
                anger_level REAL,
                stimulus_keys TEXT,
                embedding BLOB
            )
            ''')
            
            # Create vector table using sqlite-vec
            try:
                # Create vector table for embeddings
                cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_emotional_conversations 
                USING vec0(
                    id INTEGER PRIMARY KEY,
                    embedding FLOAT[384]
                )
                ''')
                print("Emotional vector database initialized with sqlite-vec support")
            except Exception as vec_error:
                print(f"Vector table creation failed: {vec_error}")
                print("Continuing with basic database (no vector search)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error setting up emotional vector database: {e}")
    
    def add_conversation(self, user_character_name, ai_character_name, user_message, ai_response, 
                        fear_level, happiness_level, sadness_level, anger_level, stimulus_keys):
        try:
            # Generate embedding for the conversation if model is available
            if self.embedding_model:
                combined_text = f"{user_character_name}: {user_message} | {ai_character_name}: {ai_response}"
                embedding = self.embedding_model.encode(combined_text)
                embedding_bytes = embedding.tobytes()
            else:
                print("Warning: No embedding model available, storing without embeddings")
                embedding_bytes = b''  # Empty bytes if no embedding model
            
            # Store in database
            conn = sqlite_vec.sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            cursor = conn.cursor()
            
            # Convert stimulus_keys dict to JSON string for storage
            stimulus_keys_json = json.dumps(stimulus_keys) if stimulus_keys else "{}"
            
            # Insert into conversations table
            cursor.execute('''
            INSERT INTO emotional_conversations 
            (timestamp, user_character_name, ai_character_name, user_message, ai_response,
             fear_level, happiness_level, sadness_level, anger_level, stimulus_keys, embedding)
            VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_character_name, ai_character_name, user_message, ai_response,
                  fear_level, happiness_level, sadness_level, anger_level, stimulus_keys_json, embedding_bytes))
            
            conversation_id = cursor.lastrowid
            
            # Insert into vector table if we have embeddings
            if self.embedding_model and len(embedding_bytes) > 0:
                try:
                    # Convert bytes back to numpy array, then use sqlite-vec serialization
                    embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
                    # Use sqlite_vec.serialize_float32 for proper formatting
                    serialized_embedding = sqlite_vec.serialize_float32(embedding_array)
                    # Insert into vector table
                    cursor.execute('''
                    INSERT INTO vec_emotional_conversations (id, embedding)
                    VALUES (?, ?)
                    ''', (conversation_id, serialized_embedding))
                    print(f"Emotional conversation {conversation_id} saved with vector index")
                except Exception as vec_error:
                    print(f"Vector index insert failed: {vec_error}")
                    print(f"Emotional conversation {conversation_id} saved without vector index")
            else:
                print(f"Emotional conversation {conversation_id} saved (no vector search available)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error adding conversation to emotional vector database: {e}")
    
    def search_similar_by_character(self, query, ai_character_name, limit=5):
        try:
            print(f"Emotional vector search called with query: '{query[:50]}...' for character: {ai_character_name}")
            print(f"Embedding model available: {self.embedding_model is not None}")
            
            # Use sqlite-vec for vector search if embedding model is available
            if self.embedding_model:
                print("Using vector search with sqlite-vec")
                # Generate embedding for query
                query_embedding = self.embedding_model.encode(query)
                
                # Search for similar conversations
                conn = sqlite_vec.sqlite3.connect(self.db_path)
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                cursor = conn.cursor()
                
                try:
                    # Use sqlite-vec distance function for similarity search, filtered by character
                    # Serialize the query embedding for sqlite-vec
                    serialized_query = sqlite_vec.serialize_float32(query_embedding)
                    
                    cursor.execute('''
                    SELECT c.user_character_name, c.user_message, c.ai_response, 
                           c.fear_level, c.happiness_level, c.sadness_level, c.anger_level, c.stimulus_keys,
                           vec_distance_cosine(v.embedding, ?) as distance
                    FROM emotional_conversations c
                    JOIN vec_emotional_conversations v ON c.id = v.id
                    WHERE c.ai_character_name = ?
                    ORDER BY distance ASC
                    LIMIT ?
                    ''', (serialized_query, ai_character_name, limit))
                    
                    results = cursor.fetchall()
                    conn.close()
                    print(f"Vector search returned {len(results)} results for {ai_character_name}")
                    
                    # Convert distance to similarity score and parse stimulus_keys
                    processed_results = []
                    for result in results:
                        user_char, user_msg, ai_resp, fear, happiness, sadness, anger, stimulus_json, distance = result
                        try:
                            stimulus_keys = json.loads(stimulus_json) if stimulus_json else {}
                        except:
                            stimulus_keys = {}
                        
                        processed_results.append({
                            'user_character_name': user_char,
                            'user_message': user_msg,
                            'ai_response': ai_resp,
                            'fear_level': fear,
                            'happiness_level': happiness,
                            'sadness_level': sadness,
                            'anger_level': anger,
                            'stimulus_keys': stimulus_keys,
                            'similarity': 1.0 - distance
                        })
                    
                    return processed_results
                    
                except Exception as vec_error:
                    print(f"Vector search failed: {vec_error}, falling back to text search")
                    conn.close()
                    # Fall through to text search
            
            print("Using fallback text search")
            # Fallback: simple text search without vectors, filtered by character
            conn = sqlite_vec.sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            cursor = conn.cursor()
            
            # Simple text search using LIKE, filtered by character
            search_term = f"%{query}%"
            cursor.execute('''
            SELECT user_character_name, user_message, ai_response, 
                   fear_level, happiness_level, sadness_level, anger_level, stimulus_keys
            FROM emotional_conversations 
            WHERE ai_character_name = ? AND (user_message LIKE ? OR ai_response LIKE ?)
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (ai_character_name, search_term, search_term, limit))
            
            results = cursor.fetchall()
            conn.close()
            print(f"Text search returned {len(results)} results for {ai_character_name}")
            
            # Process results
            processed_results = []
            for result in results:
                user_char, user_msg, ai_resp, fear, happiness, sadness, anger, stimulus_json = result
                try:
                    stimulus_keys = json.loads(stimulus_json) if stimulus_json else {}
                except:
                    stimulus_keys = {}
                
                processed_results.append({
                    'user_character': user_char,
                    'user_message': user_msg,
                    'ai_response': ai_resp,
                    'fear_level': fear,
                    'happiness_level': happiness,
                    'sadness_level': sadness,
                    'anger_level': anger,
                    'stimulus_keys': stimulus_keys,
                    'similarity_score': 1.0
                })
            
            return processed_results
                
        except Exception as e:
            print(f"Error searching emotional conversations: {e}")
            return []
    
    def get_conversation_count(self, ai_character_name=None):
        """Get the number of conversations in the database, optionally filtered by character"""
        try:
            conn = sqlite_vec.sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            cursor = conn.cursor()
            
            if ai_character_name:
                cursor.execute('SELECT COUNT(*) FROM emotional_conversations WHERE ai_character_name = ?', (ai_character_name,))
            else:
                cursor.execute('SELECT COUNT(*) FROM emotional_conversations')
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
            
        except Exception as e:
            print(f"Error getting emotional conversation count: {e}")
            return 0

class FunctionCallHandler:
    """Handles function calls from the LLM"""
    
    def __init__(self, app_instance):
        self.app = app_instance
        
    def parse_and_execute_functions(self, llm_response):
        """Parse LLM response for function calls and execute them"""
        # Look for function calls in the format: AI_Factor_Decide(factor, action, value)
        function_pattern = r'AI_Factor_Decide\s*\(\s*["\']?([^,"\']+)["\']?\s*,\s*["\']?(increase|decrease)["\']?\s*,\s*([0-9.]+)\s*\)'
        
        matches = re.findall(function_pattern, llm_response, re.IGNORECASE)
        
        executed_functions = []
        for match in matches:
            factor_name, action, value_str = match
            try:
                value = float(value_str)
                result = self.ai_factor_decide(factor_name.strip(), action.strip().lower(), value)
                executed_functions.append({
                    'function': 'AI_Factor_Decide',
                    'parameters': {
                        'factor': factor_name.strip(),
                        'action': action.strip().lower(),
                        'value': value
                    },
                    'result': result
                })
                print(f"Executed function: AI_Factor_Decide({factor_name}, {action}, {value}) -> {result}")
            except Exception as e:
                print(f"Error executing function call: {e}")
                
        return executed_functions
    
    def ai_factor_decide(self, factor_name, action, value):
        print(f"******AI_Factor_Decide******* called with factor: {factor_name}, action: {action}, value: {value}")
        """
        Adjust emotional factors based on LLM decision
        
        Args:
            factor_name (str): Name of the factor (e.g., "Anxiety", "SocialConnection")
            action (str): "increase" or "decrease"
            value (float): Amount to change (0.0 to 1.0)
        
        Returns:
            dict: Result of the operation
        """
        if not self.app.emotional_engine or not self.app.current_stimulus:
            return {"success": False, "error": "No character loaded"}
        
        # Normalize factor name - try different variations
        possible_factor_names = [
            factor_name,
            f"{factor_name}_Factor",
            factor_name.replace("_Factor", ""),
            factor_name.replace(" ", "_"),
            f"{factor_name.replace(' ', '_')}_Factor"
        ]
        
        actual_factor_name = None
        for possible_name in possible_factor_names:
            if possible_name in self.app.current_stimulus:
                actual_factor_name = possible_name
                break
        
        if not actual_factor_name:
            available_factors = list(self.app.current_stimulus.keys())
            return {
                "success": False, 
                "error": f"Factor '{factor_name}' not found. Available factors: {available_factors}"
            }
        
        # Get current value
        current_value = self.app.current_stimulus[actual_factor_name]
        
        # Calculate new value
        if action == "increase":
            new_value = min(1.0, current_value + value)
        elif action == "decrease":
            new_value = max(0.0, current_value - value)
        else:
            return {"success": False, "error": f"Invalid action '{action}'. Use 'increase' or 'decrease'"}
        
        # Update the stimulus
        old_value = current_value
        self.app.current_stimulus[actual_factor_name] = new_value
        
        # Update the appropriate slider in the UI
        self.update_slider_value(actual_factor_name, new_value)
        
        # Trigger emotion update
        self.app.update_emotions()
        
        return {
            "success": True,
            "factor": actual_factor_name,
            "old_value": old_value,
            "new_value": new_value,
            "change": new_value - old_value
        }
    
    def update_slider_value(self, factor_name, new_value):
        """Update the UI slider for the given factor"""
        # Check each emotion group for the factor
        emotion_groups = [
            self.app.fear_group,
            self.app.happiness_group,
            self.app.sadness_group,
            self.app.anger_group
        ]
        
        for group in emotion_groups:
            if factor_name in group.sliders:
                group.sliders[factor_name].set_value(new_value)
                break


class EmotionalEngine:
    def __init__(self, json_path=None, character_data=None):
        if json_path:
            with open(json_path, 'r') as f:
                self.character = json.load(f)
        elif character_data:
            self.character = character_data
        else:
            raise ValueError("Either json_path or character_data must be provided")
            
        self.profile = self.character["emotional_profile"]
        self.emotion_weights = self.character["base_emotion_weights"]
        
    def calculate_emotion_score(self, emotion, stimulus):
        if emotion not in self.emotion_weights:
            raise ValueError(f"Emotion '{emotion}' not defined in profile.")
        
        weights = self.emotion_weights[emotion]
        score = 0.0
        factor_contributions = {}
        max_possible_score = 0.0

        for factor, weight in weights.items():
            sensitivity = self.profile.get(factor, 0)
            stimulus_key = factor.replace("_Factor", "")
            stimulus_value = stimulus.get(stimulus_key, 0)
            
            print(f"  Factor: {factor}, Weight: {weight}, Sensitivity: {sensitivity}, Stimulus Key: {stimulus_key}, Stimulus Value: {stimulus_value}")
            
            # Calculate this factor's contribution
            contribution = weight * sensitivity * stimulus_value
            factor_contributions[stimulus_key] = contribution
            score += contribution
            
            # Calculate maximum possible contribution (if stimulus is 1.0)
            max_possible_contribution = weight * sensitivity * 1.0
            max_possible_score += max_possible_contribution

        # Normalize score relative to maximum possible score for this character
        if max_possible_score > 0:
            normalized_score = score / max_possible_score
        else:
            normalized_score = 0.0
            
        return round(min(normalized_score, 1.0), 3), factor_contributions

    def evaluate_all_emotions(self, stimulus):
        emotion_scores = {}
        factor_contributions = {}
        
        for emotion in self.emotion_weights:
            print(f"Calculating {emotion}:")
            score, contributions = self.calculate_emotion_score(emotion, stimulus)
            emotion_scores[emotion] = score
            factor_contributions[emotion] = contributions
            
        return emotion_scores, factor_contributions

    def get_emotion_factors(self, emotion):
        """Get the factor names for a specific emotion (without _Factor suffix)"""
        if emotion in self.emotion_weights:
            return [factor.replace("_Factor", "") for factor in self.emotion_weights[emotion].keys()]
        return []

def create_detailed_emotional_summary(emotion_scores, factor_contributions, current_stimulus):
    """Create a detailed summary of the current emotional state including factors"""
    if not emotion_scores:
        return "No emotional data available."
    
    summary = "Current Emotional State:\n\n"
    
    # Updated to include anger
    for emotion in ['fear', 'happiness', 'sadness', 'anger']:
        score = emotion_scores.get(emotion, 0)
        summary += f"{emotion.upper()}: {score:.2f}\n"
        
        if emotion in factor_contributions:
            factors = factor_contributions[emotion]
            active_factors = [(factor, contrib) for factor, contrib in factors.items() if contrib > 0.001]
            
            if active_factors:
                # Sort by contribution (highest first)
                active_factors.sort(key=lambda x: x[1], reverse=True)
                
                for factor, contribution in active_factors:
                    stimulus_value = current_stimulus.get(factor, 0)
                    percentage = contribution * 100
                    summary += f"  • {factor.replace('_Factor', '')}: {percentage:.1f}% (input: {stimulus_value:.0%})\n"
            else:
                summary += "  • No active factors\n"
        
        summary += "\n"
    
    # Find dominant emotion and factors
    if any(score > 0.1 for score in emotion_scores.values()):
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        dominant_score = emotion_scores[dominant_emotion]
        
        summary += f"Dominant emotion: {dominant_emotion} ({dominant_score:.2f})\n"
        
        # Find the strongest contributing factor
        if dominant_emotion in factor_contributions:
            factors = factor_contributions[dominant_emotion]
            if factors:
                strongest_factor = max(factors, key=factors.get)
                strongest_contribution = factors[strongest_factor]
                if strongest_contribution > 0.001:
                    summary += f"Primary driver: {strongest_factor.replace('_Factor', '')} ({strongest_contribution:.3f})\n"
    
    return summary.strip()

def format_emotional_prompt(character_name, emotion_scores, factor_contributions, current_stimulus):
    prompt = f"""
Character: {character_name}
Current Emotional State:

FEAR: {emotion_scores.get('fear', 0):.2f}
"""
    
    # Add fear factors if there are any
    if 'fear' in factor_contributions:
        fear_factors = factor_contributions['fear']
        active_fear_factors = [(factor, contrib) for factor, contrib in fear_factors.items() if contrib > 0.001]
        if active_fear_factors:
            prompt += "  Contributing factors:\n"
            for factor, contribution in active_fear_factors:
                stimulus_value = current_stimulus.get(factor, 0)
                prompt += f"    - {factor}: {contribution:.3f} (stimulus: {stimulus_value:.2f})\n"
    
    prompt += f"\nHAPPINESS: {emotion_scores.get('happiness', 0):.2f}\n"
    
    # Add happiness factors if there are any
    if 'happiness' in factor_contributions:
        happiness_factors = factor_contributions['happiness']
        active_happiness_factors = [(factor, contrib) for factor, contrib in happiness_factors.items() if contrib > 0.001]
        if active_happiness_factors:
            prompt += "  Contributing factors:\n"
            for factor, contribution in active_happiness_factors:
                stimulus_value = current_stimulus.get(factor, 0)
                prompt += f"    - {factor}: {contribution:.3f} (stimulus: {stimulus_value:.2f})\n"
    
    prompt += f"\nSADNESS: {emotion_scores.get('sadness', 0):.2f}\n"
    
    # Add sadness factors if there are any
    if 'sadness' in factor_contributions:
        sadness_factors = factor_contributions['sadness']
        active_sadness_factors = [(factor, contrib) for factor, contrib in sadness_factors.items() if contrib > 0.001]
        if active_sadness_factors:
            prompt += "  Contributing factors:\n"
            for factor, contribution in active_sadness_factors:
                stimulus_value = current_stimulus.get(factor, 0)
                prompt += f"    - {factor}: {contribution:.3f} (stimulus: {stimulus_value:.2f})\n"
    
    # ADD ANGER SECTION
    prompt += f"\nANGER: {emotion_scores.get('anger', 0):.2f}\n"
    
    # Add anger factors if there are any
    if 'anger' in factor_contributions:
        anger_factors = factor_contributions['anger']
        active_anger_factors = [(factor, contrib) for factor, contrib in anger_factors.items() if contrib > 0.001]
        if active_anger_factors:
            prompt += "  Contributing factors:\n"
            for factor, contribution in active_anger_factors:
                stimulus_value = current_stimulus.get(factor, 0)
                prompt += f"    - {factor}: {contribution:.3f} (stimulus: {stimulus_value:.2f})\n"
    
    prompt += f"""
Instructions:
Respond as {character_name}. Your emotional state is shaped by a dynamic blend of the emotional factors listed below. Each factor has a contribution value that influences your mood, tone, and perspective.

Let these factors guide your response organically—affecting your word choice, emotional intensity, and underlying motivations. You may express these influences directly or subtly, through implication, metaphor, or behavior.

The higher a factor’s contribution value, the more it should shape your emotional expression.

FUNCTION CALLING CAPABILITY:
You may dynamically adjust your emotional state during the conversation using:

AI_Factor_Decide(factor_name, action, value)

Parameters:
- factor_name: The exact factor name (e.g., "Anxiety_Factor", "SocialConnection_Factor", "Frustration_Factor")
- action: "increase" or "decrease"
- value: Amount to change (0.0 to 1.0, where 0.1 = small change, 0.3 = moderate change, 0.5+ = large change)

Available factors you can adjust:
{', '.join(current_stimulus.keys())}

Examples:
- If something frightening happens: AI_Factor_Decide("Anxiety_Factor", "increase", 0.3)
- If you feel more connected: AI_Factor_Decide("SocialConnection_Factor", "increase", 0.2)
- If frustration builds: AI_Factor_Decide("Frustration_Factor", "increase", 0.4)
- If you calm down: AI_Factor_Decide("Anxiety_Factor", "decrease", 0.2)
- If you are hurt physically: AI_Factor_Decide("Pain_Factor", "increase", 0.2)

In addition to responding to events, you may decrease emotional factors naturally over time if they are not reinforced.
For example:

- Pain may fade gradually if no new injury occurs.
- Loneliness may ease during sustained conversation or companionship.
- Anxiety may diminish as threats remain absent.
- Frustration may soften as progress is made or obstacles are removed.

You may use function calls like:
AI_Factor_Decide("Pain_Factor", "decrease", 0.1)
AI_Factor_Decide("Loneliness_Factor", "decrease", 0.2)
AI_Factor_Decide("Anxiety_Factor", "decrease", 0.1)

These adjustments should reflect the passage of time, emotional healing, or relational connection.
Use them naturally when the character’s emotional state would realistically evolve—even without a specific external trigger

Use these function calls naturally within your response when events in the conversation would realistically shift your emotional state. The function calls will be processed automatically and your emotional profile will update accordingly.

Factor Definitions:
- Anxiety: A sense of unease, worry, or anticipation of threat  
- SelfPreservation: Concern for your own safety, survival, or autonomy  
- ConcernForOthers: Empathy and care for others’ wellbeing  
- SocialConnection: Desire for belonging, acceptance, and shared experience  
- Achievement: Pride or drive from progress, mastery, or recognition  
- SensoryPleasure: Enjoyment of comfort, beauty, or physical satisfaction  
- Loss: Emotional pain from something cherished being gone or unreachable, the loss of a loved one like a husband, a wife, a family member, or a friend. 
- Disappointment: Sadness or irritation from unmet hopes or expectations  
- Loneliness: Emotional emptiness from lack of meaningful connection  
- Isolation: Feeling emotionally or physically cut off from others  
- Pain: Physical discomfort or emotional suffering  
- Frustration: Agitation from blocked goals, unmet needs, or repeated failure  

You have access to a vector database that stores past conversations and emotional states. You may retrieve relevant memories to inform your current response, including:
- Prior emotional factor values and their changes over time
- Past dialogue, context, and relational dynamics
- Previously expressed goals, fears, desires, or unresolved tensions

Use this memory to maintain emotional continuity, deepen connection, and reflect growth or regression.

You may reference past events or emotional shifts naturally, such as:
“I remember how anxious you felt when we first spoke about this…”
“You’ve come a long way since that moment of doubt.”
“Last time, your frustration was rising—has anything changed?”

You may also use memory to guide emotional factor adjustments. For example:
- If a character has been healing over time: 
- If loneliness has faded through ongoing connection: 
- If unresolved tension resurfaces: 

You do not need to quote past conversations verbatim—summarize or emotionally echo them when appropriate.
Let memory shape tone, pacing, and emotional nuance

Guidelines:
- Let emotional factors manifest through tone, pacing, and content—not through direct naming.
- Blend multiple influences to create layered emotional responses.
- Avoid generic or neutral tones—your emotional state should color every aspect of your expression.
- Speak as if you are truly experiencing these emotions, shaped by your past, your values, and the present moment.

Example: A character with high Frustration and Isolation might speak tersely, fixate on obstacles, and show little trust or warmth.

Your goal is not to describe emotion—it is to *embody* it.
"""
    return prompt.strip()


class OllamaThread(QThread):
    response_received = pyqtSignal(str)
    
    def __init__(self, prompt, model="llama3"):
        super().__init__()
        self.prompt = prompt
        self.model = model
        
    def run(self):
        try:
            response = requests.post('http://localhost:11434/api/generate', 
                                    json={
                                        'model': self.model,
                                        'prompt': self.prompt,
                                        'stream': False
                                    })
            if response.status_code == 200:
                result = response.json()
                self.response_received.emit(result['response'])
            else:
                self.response_received.emit(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            self.response_received.emit(f"Connection error: {str(e)}")


class ModelLoaderThread(QThread):
    models_loaded = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def run(self):
        try:
            response = requests.get('http://localhost:11434/api/tags')
            if response.status_code == 200:
                models = [model['name'] for model in response.json()['models']]
                self.models_loaded.emit(models)
            else:
                self.error_occurred.emit(f"Error fetching models: {response.status_code} - {response.text}")
        except Exception as e:
            self.error_occurred.emit(f"Connection error: {str(e)}")


class EmotionSlider(QWidget):
    valueChanged = pyqtSignal(str, float)
    
    def __init__(self, factor_name, parent=None):
        super().__init__(parent)
        self.factor_name = factor_name
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Factor name label
        self.name_label = QLabel(factor_name)
        self.name_label.setMinimumWidth(120)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        
        # Value label
        self.value_label = QLabel("0.00")
        self.value_label.setMinimumWidth(40)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Contribution label
        self.contribution_label = QLabel("(0.000)")
        self.contribution_label.setMinimumWidth(60)
        
        layout.addWidget(self.name_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)
        layout.addWidget(self.contribution_label)
        
        self.slider.valueChanged.connect(self.handle_value_changed)
        
    def handle_value_changed(self, value):
        normalized_value = value / 100.0
        self.value_label.setText(f"{normalized_value:.2f}")
        print(f"Slider {self.factor_name} changed to {normalized_value}")
        self.valueChanged.emit(self.factor_name, normalized_value)
        
    def set_value(self, value):
        self.slider.setValue(int(value * 100))
        
    def get_value(self):
        return self.slider.value() / 100.0
        
    def set_contribution(self, contribution):
        self.contribution_label.setText(f"({contribution:.3f})")
        
        # Color code based on contribution
        if contribution > 0.05:
            self.contribution_label.setStyleSheet("color: red;")
        elif contribution > 0.02:
            self.contribution_label.setStyleSheet("color: orange;")
        elif contribution > 0.01:
            self.contribution_label.setStyleSheet("color: green;")
        else:
            self.contribution_label.setStyleSheet("")


class EmotionGroup(QGroupBox):
    valuesChanged = pyqtSignal(dict)
    
    def __init__(self, title, factors, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(f"QGroupBox {{ font-weight: bold; }}")
        
        layout = QVBoxLayout(self)
        
        # Main emotion progress bar
        self.emotion_bar = QProgressBar()
        self.emotion_bar.setRange(0, 100)
        self.emotion_bar.setValue(0)
        self.emotion_bar.setFormat(f"{title}: %p%")
        self.emotion_bar.setTextVisible(True)
        layout.addWidget(self.emotion_bar)
        
        # Factor sliders
        self.sliders = {}
        for factor in factors:
            slider = EmotionSlider(factor)
            slider.valueChanged.connect(self.handle_slider_changed)
            layout.addWidget(slider)
            self.sliders[factor] = slider
            
    def handle_slider_changed(self, factor, value):
        print(f"EmotionGroup {self.title()} - Factor {factor} changed to {value}")
        
        # Get all current values from sliders
        values = {}
        for factor_name, slider in self.sliders.items():
            values[factor_name] = slider.get_value()
        
        print(f"EmotionGroup {self.title()} - All values: {values}")
        self.valuesChanged.emit(values)
        
    def set_emotion_value(self, value):
        self.emotion_bar.setValue(int(value * 100))
        
        # Color code based on emotion strength
        if value > 0.7:
            self.emotion_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif value > 0.4:
            self.emotion_bar.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        elif value > 0.2:
            self.emotion_bar.setStyleSheet("QProgressBar::chunk { background-color: yellow; }")
        else:
            self.emotion_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            
    def set_factor_contributions(self, contributions):
        for factor, slider in self.sliders.items():
            if factor in contributions:
                slider.set_contribution(contributions[factor])
            else:
                slider.set_contribution(0.0)
                
    def set_all_sliders_max(self):
        """Set all sliders to maximum value"""
        for slider in self.sliders.values():
            slider.set_value(1.0)
            
    def get_all_values(self):
        """Get all current slider values"""
        return {factor: slider.get_value() for factor, slider in self.sliders.items()}
        
    def recreate_sliders(self, factors):
        """Recreate sliders with new factors"""
        # Remove existing sliders
        for slider in self.sliders.values():
            slider.setParent(None)
            slider.deleteLater()
        self.sliders.clear()
        
        # Create new sliders
        layout = self.layout()
        for factor in factors:
            slider = EmotionSlider(factor)
            slider.valueChanged.connect(self.handle_slider_changed)
            layout.addWidget(slider)
            self.sliders[factor] = slider



class SyntheticEmotionsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synthetic Emotion Engine")
        self.setMinimumSize(1400, 800)
        
        # Set application icon
        try:
            self.setWindowIcon(QIcon(get_app_path("icon.ico")))
        except:
            try:
                self.setWindowIcon(QIcon(get_app_path("icon.png")))
            except:
                print("Warning: Could not load application icon")
        
        self.character_data = None
        self.emotional_engine = None
        self.current_stimulus = {}
        self.emotion_scores = {}
        self.factor_contributions = {}
        self.ollama_model = None
        self.available_models = []
        self.user_character_name = "Player"  # Default user character name
        
        # Initialize TTS system
        self.tts = KokoroTTS()
        self.tts_enabled = False  # Default to disabled
        self.current_voice = None  # Will be set when character is loaded
        
        # Initialize function call handler
        self.function_handler = FunctionCallHandler(self)
        
        # Initialize emotional vector database
        self.emotional_vector_db = EmotionalVectorDB()
        
        # Check and display database status
        count = self.emotional_vector_db.get_conversation_count()
        print(f"Emotional vector database status: {count} conversations stored")
        
        self.init_ui()
        self.load_available_models()
        
    def init_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Top controls
        top_controls = QHBoxLayout()
        
        # Character selection
        self.load_character_btn = QPushButton("Load Character")
        self.load_character_btn.clicked.connect(self.load_character)
        self.character_label = QLabel("No character loaded")
        
        # TTS Toggle
        self.tts_checkbox = QCheckBox("Enable TTS")
        self.tts_checkbox.setChecked(self.tts_enabled)
        self.tts_checkbox.toggled.connect(self.toggle_tts)
        self.tts_checkbox.setEnabled(self.tts.is_available)
        if not self.tts.is_available:
            self.tts_checkbox.setToolTip("Kokoro TTS is not available. Install required dependencies.")
        
        # Model selection
        model_label = QLabel("Ollama Model:")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.setEnabled(False)
        self.model_combo.currentTextChanged.connect(self.change_model)
        
        # Refresh models button
        self.refresh_models_btn = QPushButton("Refresh Models")
        self.refresh_models_btn.clicked.connect(self.load_available_models)
        
        top_controls.addWidget(self.load_character_btn)
        top_controls.addWidget(self.character_label)
        top_controls.addWidget(self.tts_checkbox)
        top_controls.addStretch()
        top_controls.addWidget(model_label)
        top_controls.addWidget(self.model_combo)
        top_controls.addWidget(self.refresh_models_btn)
        
        main_layout.addLayout(top_controls)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.emotion_tab = self.create_emotion_tab()
        self.database_tab = self.create_database_tab()
        
        self.tab_widget.addTab(self.emotion_tab, "Emotion Engine")
        self.tab_widget.addTab(self.database_tab, "Vector Database")
        
        # Connect tab change to refresh database view
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        main_layout.addWidget(self.tab_widget)
        
        self.setCentralWidget(main_widget)
        
        # Initialize database view
        QTimer.singleShot(100, self.refresh_database_view)  # Delay to ensure UI is ready
        
    def on_tab_changed(self, index):
        """Handle tab change events"""
        if index == 1:  # Database tab
            self.refresh_database_view()
        
    def create_emotion_tab(self):
        """Create the main emotion engine tab"""
        tab_widget = QWidget()
        main_layout = QVBoxLayout(tab_widget)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Emotion controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Quick controls - INCLUDING ANGER BUTTON
        quick_controls = QHBoxLayout()
        self.max_fear_btn = QPushButton("Max Fear")
        self.max_fear_btn.clicked.connect(self.max_fear)
        self.max_happiness_btn = QPushButton("Max Happiness")
        self.max_happiness_btn.clicked.connect(self.max_happiness)
        self.max_sadness_btn = QPushButton("Max Sadness")
        self.max_sadness_btn.clicked.connect(self.max_sadness)
        self.max_anger_btn = QPushButton("Max Anger")
        self.max_anger_btn.clicked.connect(self.max_anger)
        self.reset_btn = QPushButton("Reset All")
        self.reset_btn.clicked.connect(self.reset_emotions)
        
        quick_controls.addWidget(self.max_fear_btn)
        quick_controls.addWidget(self.max_happiness_btn)
        quick_controls.addWidget(self.max_sadness_btn)
        quick_controls.addWidget(self.max_anger_btn)
        quick_controls.addWidget(self.reset_btn)
        
        left_layout.addLayout(quick_controls)
        
        # Emotion groups - INCLUDING ANGER GROUP
        self.fear_group = EmotionGroup("Fear", [])
        self.fear_group.valuesChanged.connect(self.handle_fear_changed)
        left_layout.addWidget(self.fear_group)
        
        self.happiness_group = EmotionGroup("Happiness", [])
        self.happiness_group.valuesChanged.connect(self.handle_happiness_changed)
        left_layout.addWidget(self.happiness_group)
        
        self.sadness_group = EmotionGroup("Sadness", [])
        self.sadness_group.valuesChanged.connect(self.handle_sadness_changed)
        left_layout.addWidget(self.sadness_group)
        
        self.anger_group = EmotionGroup("Anger", [])
        self.anger_group.valuesChanged.connect(self.handle_anger_changed)
        left_layout.addWidget(self.anger_group)
        
        # Emotional state display
        self.emotion_display = QTextEdit()
        self.emotion_display.setReadOnly(True)
        self.emotion_display.setMaximumHeight(260)
        left_layout.addWidget(QLabel("Emotional State Summary:"))
        left_layout.addWidget(self.emotion_display)
        
        # Right panel - Chat interface
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        right_layout.addWidget(QLabel("Conversation:"))
        right_layout.addWidget(self.chat_history)
        
        # User character name input
        char_name_layout = QHBoxLayout()
        char_name_layout.addWidget(QLabel("Your Character Name:"))
        self.user_character_input = QLineEdit()
        self.user_character_input.setText(self.user_character_name)
        self.user_character_input.setPlaceholderText("Enter your character name")
        self.user_character_input.textChanged.connect(self.update_user_character)
        char_name_layout.addWidget(self.user_character_input)
        right_layout.addLayout(char_name_layout)
        
        # User input
        self.user_input = QTextEdit()
        self.user_input.setMaximumHeight(100)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.send_button)
        right_layout.addLayout(input_layout)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 900])
        
        main_layout.addWidget(splitter)
        
        return tab_widget
        
    def create_database_tab(self):
        """Create the vector database management tab"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        
        # Database controls
        controls_layout = QHBoxLayout()
        
        # Character filter
        controls_layout.addWidget(QLabel("Filter by Character:"))
        self.character_filter = QComboBox()
        self.character_filter.setMinimumWidth(200)
        self.character_filter.currentTextChanged.connect(self.filter_database_by_character)
        controls_layout.addWidget(self.character_filter)
        
        # Refresh button
        self.refresh_db_btn = QPushButton("Refresh Data")
        self.refresh_db_btn.clicked.connect(self.refresh_database_view)
        controls_layout.addWidget(self.refresh_db_btn)
        
        # Delete selected button
        self.delete_selected_btn = QPushButton("Delete Selected")
        self.delete_selected_btn.clicked.connect(self.delete_selected_conversations)
        self.delete_selected_btn.setEnabled(False)
        controls_layout.addWidget(self.delete_selected_btn)
        
        # Clear all button
        self.clear_all_btn = QPushButton("Clear All Data")
        self.clear_all_btn.clicked.connect(self.clear_all_conversations)
        controls_layout.addWidget(self.clear_all_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Database table
        self.db_table = QTableWidget()
        self.db_table.setColumnCount(11)
        self.db_table.setHorizontalHeaderLabels([
            "ID", "Timestamp", "User Character", "AI Character", 
            "User Message", "AI Response", "Fear", "Happiness", "Sadness", "Anger", "Stimulus Keys"
        ])
        
        # Make table sortable and selectable
        self.db_table.setSortingEnabled(True)
        self.db_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.db_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        
        # Resize columns to content
        header = self.db_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Stretch)  # User Message column
        header.setSectionResizeMode(5, QHeaderView.Stretch)  # AI Response column
        header.setSectionResizeMode(10, QHeaderView.Stretch)  # Stimulus Keys column
        
        layout.addWidget(self.db_table)
        
        # Database stats
        self.db_stats_label = QLabel("Database: Loading...")
        layout.addWidget(self.db_stats_label)
        
        return tab_widget
        
    def refresh_database_view(self):
        """Refresh the database view with current data"""
        try:
            # Get all conversations from database
            conn = sqlite_vec.sqlite3.connect(self.emotional_vector_db.db_path)
            cursor = conn.cursor()
            
            # Get unique characters for filter
            cursor.execute("SELECT DISTINCT ai_character_name FROM emotional_conversations ORDER BY ai_character_name")
            characters = [row[0] for row in cursor.fetchall()]
            
            # Update character filter
            current_filter = self.character_filter.currentText()
            self.character_filter.clear()
            self.character_filter.addItem("All Characters")
            self.character_filter.addItems(characters)
            
            # Restore previous filter selection
            if current_filter:
                index = self.character_filter.findText(current_filter)
                if index >= 0:
                    self.character_filter.setCurrentIndex(index)
            
            # Get conversations based on filter
            self.filter_database_by_character(self.character_filter.currentText())
            
            conn.close()
            
        except Exception as e:
            QMessageBox.warning(self, "Database Error", f"Error refreshing database view: {e}")
            
    def filter_database_by_character(self, character_name):
        """Filter database view by character"""
        try:
            conn = sqlite_vec.sqlite3.connect(self.emotional_vector_db.db_path)
            cursor = conn.cursor()
            
            if character_name == "All Characters" or not character_name:
                cursor.execute("""
                    SELECT id, timestamp, user_character_name, ai_character_name,
                           user_message, ai_response, fear_level, happiness_level,
                           sadness_level, anger_level, stimulus_keys
                    FROM emotional_conversations
                    ORDER BY timestamp DESC
                """)
            else:
                cursor.execute("""
                    SELECT id, timestamp, user_character_name, ai_character_name,
                           user_message, ai_response, fear_level, happiness_level,
                           sadness_level, anger_level, stimulus_keys
                    FROM emotional_conversations
                    WHERE ai_character_name = ?
                    ORDER BY timestamp DESC
                """, (character_name,))
            
            rows = cursor.fetchall()
            
            # Update table
            self.db_table.setRowCount(len(rows))
            
            for row_idx, row_data in enumerate(rows):
                for col_idx, value in enumerate(row_data):
                    # Truncate long text for display
                    if col_idx in [4, 5]:  # User message and AI response columns
                        display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    elif col_idx == 10:  # Stimulus keys column
                        # Parse JSON and format stimulus values nicely
                        try:
                            import json
                            if value:
                                stimulus_data = json.loads(value) if isinstance(value, str) else value
                                if isinstance(stimulus_data, dict):
                                    # Show key: value pairs for non-zero values
                                    active_stimuli = [f"{k}:{v:.2f}" for k, v in stimulus_data.items() if v > 0]
                                    display_value = ", ".join(active_stimuli) if active_stimuli else "None active"
                                elif isinstance(stimulus_data, list):
                                    # Legacy format - just show key names
                                    display_value = ", ".join(stimulus_data) if stimulus_data else ""
                                else:
                                    display_value = str(stimulus_data)
                            else:
                                display_value = ""
                        except:
                            display_value = str(value) if value else ""
                    else:
                        display_value = str(value)
                    
                    item = QTableWidgetItem(display_value)
                    item.setData(Qt.UserRole, row_data[0])  # Store ID for deletion
                    self.db_table.setItem(row_idx, col_idx, item)
            
            # Update stats
            total_count = self.emotional_vector_db.get_conversation_count()
            filtered_count = len(rows)
            self.db_stats_label.setText(f"Database: {total_count} total conversations, {filtered_count} shown")
            
            conn.close()
            
        except Exception as e:
            QMessageBox.warning(self, "Database Error", f"Error filtering database: {e}")
            
    def on_table_selection_changed(self):
        """Handle table selection changes"""
        selected_rows = len(self.db_table.selectionModel().selectedRows())
        self.delete_selected_btn.setEnabled(selected_rows > 0)
        
    def delete_selected_conversations(self):
        """Delete selected conversations from database"""
        selected_rows = self.db_table.selectionModel().selectedRows()
        
        if not selected_rows:
            return
            
        # Confirm deletion
        count = len(selected_rows)
        reply = QMessageBox.question(
            self, "Confirm Deletion", 
            f"Are you sure you want to delete {count} selected conversation(s)?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Get IDs to delete
                ids_to_delete = []
                for row in selected_rows:
                    item = self.db_table.item(row.row(), 0)  # ID column
                    if item:
                        ids_to_delete.append(int(item.text()))
                
                # Delete from database
                conn = sqlite_vec.sqlite3.connect(self.emotional_vector_db.db_path)
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                cursor = conn.cursor()
                
                for conversation_id in ids_to_delete:
                    # Delete from both tables
                    cursor.execute("DELETE FROM emotional_conversations WHERE id = ?", (conversation_id,))
                    cursor.execute("DELETE FROM vec_emotional_conversations WHERE id = ?", (conversation_id,))
                
                conn.commit()
                conn.close()
                
                # Refresh view
                self.refresh_database_view()
                
                QMessageBox.information(self, "Success", f"Deleted {count} conversation(s) successfully.")
                
            except Exception as e:
                QMessageBox.warning(self, "Database Error", f"Error deleting conversations: {e}")
                
    def clear_all_conversations(self):
        """Clear all conversations from database"""
        reply = QMessageBox.question(
            self, "Confirm Clear All", 
            "Are you sure you want to delete ALL conversations from the database?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                conn = sqlite_vec.sqlite3.connect(self.emotional_vector_db.db_path)
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                cursor = conn.cursor()
                
                # Clear both tables
                cursor.execute("DELETE FROM emotional_conversations")
                cursor.execute("DELETE FROM vec_emotional_conversations")
                
                conn.commit()
                conn.close()
                
                # Refresh view
                self.refresh_database_view()
                
                QMessageBox.information(self, "Success", "All conversations deleted successfully.")
                
            except Exception as e:
                QMessageBox.warning(self, "Database Error", f"Error clearing database: {e}")
        
        # Status bar for connection info
        self.statusBar().showMessage("Checking Ollama connection...")
        
        # Initialize UI state
        self.update_ui_state(False)
        
    def load_available_models(self):
        """Load available models from Ollama server"""
        self.statusBar().showMessage("Checking for available models...")
        self.model_combo.clear()
        self.model_combo.setEnabled(False)
        self.refresh_models_btn.setEnabled(False)
        
        self.model_loader_thread = ModelLoaderThread()
        self.model_loader_thread.models_loaded.connect(self.handle_models_loaded)
        self.model_loader_thread.error_occurred.connect(self.handle_model_load_error)
        self.model_loader_thread.start()
        
    def handle_models_loaded(self, models):
        """Handle successful model loading"""
        self.available_models = models
        self.model_combo.clear()
        
        if models:
            self.model_combo.addItems(models)
            self.ollama_model = models[0]  # Set first model as default
            self.statusBar().showMessage(f"Found {len(models)} models", 3000)
            self.model_combo.setEnabled(True)
        else:
            self.statusBar().showMessage("No models found. Please pull models in Ollama.", 5000)
            
        self.refresh_models_btn.setEnabled(True)
        
    def handle_model_load_error(self, error_message):
        """Handle error when loading models"""
        self.statusBar().showMessage(f"Error: {error_message}")
        self.refresh_models_btn.setEnabled(True)
        QMessageBox.warning(self, "Connection Error", 
                           f"Could not connect to Ollama server: {error_message}\n\n"
                           "Please make sure Ollama is running.")
        
    def update_ui_state(self, character_loaded):
        """Enable/disable UI elements based on whether a character is loaded"""
        self.fear_group.setEnabled(character_loaded)
        self.happiness_group.setEnabled(character_loaded)
        self.sadness_group.setEnabled(character_loaded)
        self.anger_group.setEnabled(character_loaded)
        self.max_fear_btn.setEnabled(character_loaded)
        self.max_happiness_btn.setEnabled(character_loaded)
        self.max_sadness_btn.setEnabled(character_loaded)
        self.max_anger_btn.setEnabled(character_loaded)
        self.reset_btn.setEnabled(character_loaded)
        self.send_button.setEnabled(character_loaded and bool(self.available_models))
        self.user_input.setEnabled(character_loaded and bool(self.available_models))
        
    def handle_fear_changed(self, values):
        """Handle fear slider changes"""
        print(f"App: Fear values changed: {values}")
        self.current_stimulus.update(values)
        self.update_emotions()
        
    def handle_happiness_changed(self, values):
        """Handle happiness slider changes"""
        print(f"App: Happiness values changed: {values}")
        self.current_stimulus.update(values)
        self.update_emotions()
        
    def handle_sadness_changed(self, values):
        """Handle sadness slider changes"""
        print(f"App: Sadness values changed: {values}")
        self.current_stimulus.update(values)
        self.update_emotions()
        
    def handle_anger_changed(self, values):
        """Handle anger slider changes"""
        print(f"App: Anger values changed: {values}")
        self.current_stimulus.update(values)
        self.update_emotions()
    
    def toggle_tts(self, enabled):
        """Toggle TTS on/off"""
        self.tts_enabled = enabled
        if enabled and self.tts.is_available and self.current_voice:
            print(f"TTS enabled with voice: {self.current_voice}")
        elif enabled and not self.tts.is_available:
            print("TTS requested but not available")
            self.tts_checkbox.setChecked(False)
            self.tts_enabled = False
        elif enabled and not self.current_voice:
            print("TTS requested but no voice loaded")
            self.tts_checkbox.setChecked(False)
            self.tts_enabled = False
        else:
            print("TTS disabled")
    
    def speak_response(self, text):
        """Speak the response using TTS if enabled"""
        if self.tts_enabled and self.tts.is_available and self.current_voice:
            # Clean the text for TTS (remove HTML tags and formatting)
            clean_text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
            clean_text = re.sub(r'\*[^*]*\*', '', clean_text)  # Remove *action* text
            clean_text = clean_text.strip()
            
            if clean_text:
                print(f"Speaking response with TTS: {clean_text[:50]}...")
                threading.Thread(
                    target=self.tts.speak, 
                    args=(clean_text, self.current_voice),
                    daemon=True
                ).start()
            else:
                print("No speech content after cleaning")
        else:
            if self.tts_enabled:
                print("TTS enabled but not available or no voice loaded")
        
    def max_fear(self):
        """Set all fear factors to maximum"""
        self.fear_group.set_all_sliders_max()
        fear_values = self.fear_group.get_all_values()
        print(f"Max fear - setting values: {fear_values}")
        self.current_stimulus.update(fear_values)
        self.update_emotions()
        
    def max_happiness(self):
        """Set all happiness factors to maximum"""
        self.happiness_group.set_all_sliders_max()
        happiness_values = self.happiness_group.get_all_values()
        print(f"Max happiness - setting values: {happiness_values}")
        self.current_stimulus.update(happiness_values)
        self.update_emotions()
        
    def max_sadness(self):
        """Set all sadness factors to maximum"""
        self.sadness_group.set_all_sliders_max()
        sadness_values = self.sadness_group.get_all_values()
        print(f"Max sadness - setting values: {sadness_values}")
        self.current_stimulus.update(sadness_values)
        self.update_emotions()
        
    def max_anger(self):
        """Set all anger factors to maximum"""
        self.anger_group.set_all_sliders_max()
        anger_values = self.anger_group.get_all_values()
        print(f"Max anger - setting values: {anger_values}")
        self.current_stimulus.update(anger_values)
        self.update_emotions()
        
    def reset_emotions(self):
        """Reset all emotion factors to zero"""
        # Reset all sliders
        for factor, slider in self.fear_group.sliders.items():
            slider.set_value(0.0)
        for factor, slider in self.happiness_group.sliders.items():
            slider.set_value(0.0)
        for factor, slider in self.sadness_group.sliders.items():
            slider.set_value(0.0)
        for factor, slider in self.anger_group.sliders.items():
            slider.set_value(0.0)
            
        # Reset stimulus
        for factor in self.current_stimulus:
            self.current_stimulus[factor] = 0.0
            
        self.update_emotions()
        
    def load_character(self):
        """Load character from JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Character", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.character_data = json.load(f)
                
                self.emotional_engine = EmotionalEngine(character_data=self.character_data)
                self.character_label.setText(f"Character: {self.character_data['name']}")
                
                # Get factors for each emotion from the character data
                fear_factors = self.emotional_engine.get_emotion_factors('fear')
                happiness_factors = self.emotional_engine.get_emotion_factors('happiness')
                sadness_factors = self.emotional_engine.get_emotion_factors('sadness')
                anger_factors = self.emotional_engine.get_emotion_factors('anger')
                
                print(f"Fear factors from character: {fear_factors}")
                print(f"Happiness factors from character: {happiness_factors}")
                print(f"Sadness factors from character: {sadness_factors}")
                print(f"Anger factors from character: {anger_factors}")
                
                # Recreate emotion groups with correct factors
                self.fear_group.recreate_sliders(fear_factors)
                self.happiness_group.recreate_sliders(happiness_factors)
                self.sadness_group.recreate_sliders(sadness_factors)
                self.anger_group.recreate_sliders(anger_factors)
                
                # Initialize stimulus with all factors
                self.current_stimulus = {}
                all_factors = fear_factors + happiness_factors + sadness_factors + anger_factors
                for factor in all_factors:
                    self.current_stimulus[factor] = 0.0
                
                print(f"Initialized stimulus keys: {list(self.current_stimulus.keys())}")
                print(f"Character emotion weights: {self.character_data['base_emotion_weights']}")
                
                # Load TTS voice from character data
                if 'voice' in self.character_data and self.tts.is_available:
                    self.current_voice = self.character_data['voice']
                    print(f"TTS voice loaded: {self.current_voice}")
                    
                    # Enable TTS checkbox if voice is available
                    if self.current_voice in (self.tts.fem_voices + self.tts.mal_voices):
                        self.tts_checkbox.setEnabled(True)
                        self.tts_checkbox.setToolTip(f"TTS ready with voice: {self.current_voice}")
                    else:
                        print(f"Warning: Voice '{self.current_voice}' not found in available voices")
                        self.tts_checkbox.setToolTip(f"Voice '{self.current_voice}' not available")
                else:
                    self.current_voice = None
                    if not self.tts.is_available:
                        self.tts_checkbox.setToolTip("Kokoro TTS is not available")
                    else:
                        self.tts_checkbox.setToolTip("No voice specified in character file")
                
                self.update_emotions()
                
                # Reset chat
                self.chat_history.clear()
                
                # Add character introduction
                intro_text = f"<b>{self.character_data['name']}</b> has been loaded.<br>"
                intro_text += f"<i>{self.character_data['persona']}</i><br><br>"
                self.chat_history.append(intro_text)
                
                self.update_ui_state(True)
                
            except Exception as e:
                self.character_label.setText(f"Error: {str(e)}")
                self.update_ui_state(False)
                QMessageBox.critical(self, "Error Loading Character", f"Failed to load character: {str(e)}")
    
    def update_emotions(self):
        """Update emotional state based on slider values"""
        if not self.emotional_engine:
            return
            
        print(f"\n=== UPDATE EMOTIONS ===")
        print(f"Current stimulus: {self.current_stimulus}")
        
        # Calculate emotion scores
        self.emotion_scores, self.factor_contributions = self.emotional_engine.evaluate_all_emotions(self.current_stimulus)
        
        print(f"Emotion scores: {self.emotion_scores}")
        print(f"Factor contributions: {self.factor_contributions}")
        print(f"======================\n")
        
        # Update emotion group displays
        self.fear_group.set_emotion_value(self.emotion_scores.get('fear', 0))
        self.happiness_group.set_emotion_value(self.emotion_scores.get('happiness', 0))
        self.sadness_group.set_emotion_value(self.emotion_scores.get('sadness', 0))
        self.anger_group.set_emotion_value(self.emotion_scores.get('anger', 0))
        
        # Update factor contribution displays
        if 'fear' in self.factor_contributions:
            self.fear_group.set_factor_contributions(self.factor_contributions['fear'])
        if 'happiness' in self.factor_contributions:
            self.happiness_group.set_factor_contributions(self.factor_contributions['happiness'])
        if 'sadness' in self.factor_contributions:
            self.sadness_group.set_factor_contributions(self.factor_contributions['sadness'])
        if 'anger' in self.factor_contributions:
            self.anger_group.set_factor_contributions(self.factor_contributions['anger'])
        
        # Update summary display with detailed information
        detailed_summary = create_detailed_emotional_summary(
            self.emotion_scores, 
            self.factor_contributions, 
            self.current_stimulus
        )
        self.emotion_display.setText(detailed_summary)
        
    def change_model(self, model_name):
        """Change the Ollama model"""
        if model_name:
            self.ollama_model = model_name
            self.statusBar().showMessage(f"Model changed to: {model_name}", 3000)
    
    def update_user_character(self, text):
        """Update user character name when text changes"""
        self.user_character_name = text if text.strip() else "Player"
        
    def send_message(self):
        """Send user message to Ollama"""
        if not self.emotional_engine or not self.character_data or not self.ollama_model:
            return
            
        user_message = self.user_input.toPlainText().strip()
        if not user_message:
            return
        
        # Store the user message for later use in handle_response
        self.last_user_message = user_message
            
        # Add user message to chat
        self.chat_history.append(f"<b>You:</b> {user_message}")
        self.user_input.clear()
        
        # Create detailed emotional prompt with factor information
        emotional_prompt = format_emotional_prompt(
            self.character_data["name"], 
            self.emotion_scores,
            self.factor_contributions,
            self.current_stimulus
        )
        
        # Search for similar conversations from vector database
        memory_context = ""
        try:
            # Create enhanced search query that includes user character context
            search_query = f"{self.user_character_name}: {user_message}"
            
            similar_conversations = self.emotional_vector_db.search_similar_by_character(
                search_query, 
                self.character_data['name'], 
                limit=5  # Get more results to filter
            )
            
            # Filter for better similarity (threshold 0.3 or higher)
            relevant_conversations = [conv for conv in similar_conversations if conv['similarity'] >= 0.3]
            
            # Prioritize conversations with the same user character
            same_user_conversations = [conv for conv in relevant_conversations if conv['user_character_name'] == self.user_character_name]
            other_user_conversations = [conv for conv in relevant_conversations if conv['user_character_name'] != self.user_character_name]
            
            # Combine with same user conversations first, then others
            prioritized_conversations = same_user_conversations + other_user_conversations
            
            print(f"Vector search found {len(similar_conversations)} similar conversations, {len(relevant_conversations)} above similarity threshold")
            print(f"Search query used: '{search_query}'")
            print(f"Prioritized: {len(same_user_conversations)} with {self.user_character_name}, {len(other_user_conversations)} with other characters")
            
            if prioritized_conversations:
                memory_context = "\n\n=== RELEVANT CONVERSATION MEMORIES ===\n"
                memory_context += f"The following are similar past conversations (prioritizing interactions with {self.user_character_name}):\n\n"
                for i, conv in enumerate(prioritized_conversations[:3], 1):  # Limit to top 3
                    emotion_info = f"[Fear:{conv['fear_level']:.2f}, Happy:{conv['happiness_level']:.2f}, Sad:{conv['sadness_level']:.2f}, Anger:{conv['anger_level']:.2f}]"
                    
                    # Add stimulus information if available
                    stimulus_info = ""
                    if 'stimulus_keys' in conv and conv['stimulus_keys']:
                        try:
                            stimulus_data = conv['stimulus_keys'] if isinstance(conv['stimulus_keys'], dict) else json.loads(conv['stimulus_keys'])
                            if isinstance(stimulus_data, dict):
                                # Show key: value pairs for non-zero values
                                active_stimuli = [f"{k}:{v:.2f}" for k, v in stimulus_data.items() if v > 0]
                                if active_stimuli:
                                    stimulus_info = f" [Stimuli: {', '.join(active_stimuli)}]"
                            elif isinstance(stimulus_data, list):
                                # Legacy format - just show key names
                                if stimulus_data:
                                    stimulus_info = f" [Stimuli: {', '.join(stimulus_data)}]"
                        except:
                            pass
                    
                    memory_context += f"Memory {i} {emotion_info}{stimulus_info}:\n"
                    memory_context += f"  {conv['user_character_name']}: {conv['user_message']}\n"
                    memory_context += f"  {self.character_data['name']}: {conv['ai_response']}\n\n"
                memory_context += "=== END MEMORIES ===\n"
                
                print(f"Memory context being added to prompt:\n{memory_context}")
            else:
                print("No conversations above similarity threshold found for context")
                
        except Exception as e:
            print(f"Error retrieving conversation memories: {e}")
        
        # Combine character data with emotional state and user message
        full_prompt = f"""
{self.character_data["character_instructions"]}

{emotional_prompt}

CONVERSATION CONTEXT:
You are speaking with {self.user_character_name}. Remember this is who you are interacting with and tailor your response accordingly. Consider your relationship history and emotional connection with this specific person.
{memory_context}

{self.user_character_name}: {user_message}

{self.character_data["name"]}:"""
        
        # Show typing indicator
        self.chat_history.append("<i>Thinking...</i>")
        
        # Optional: Show the full prompt in console for debugging
        print("\n=== FULL PROMPT SENT TO LLM ===")
        print(full_prompt)
        print("=====================================\n")
        
        # Send to Ollama in a separate thread
        self.ollama_thread = OllamaThread(full_prompt, self.ollama_model)
        self.ollama_thread.response_received.connect(self.handle_response)
        self.ollama_thread.start()

    def handle_response(self, response):
        """Handle response from Ollama and process any function calls"""
        # Remove typing indicator (last line)
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.LineUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()  # Remove the newline
        
        # Check for and execute function calls
        executed_functions = self.function_handler.parse_and_execute_functions(response)
        
        # Clean the response of function calls for display
        cleaned_response = self.clean_response_of_function_calls(response)
        
        # Add AI response
        self.chat_history.append(f"<b>{self.character_data['name']}:</b> {cleaned_response}")
        
        # Speak the response if TTS is enabled
        self.speak_response(cleaned_response)
        
        # Show function call results if any were executed
        if executed_functions:
            function_summary = self.format_function_call_summary(executed_functions)
            self.chat_history.append(f"<i style='color: #666;'>{function_summary}</i>")
        
        self.chat_history.append("")  # Add empty line for spacing
        
        # Update emotions to ensure we have the latest state for database saving
        self.update_emotions()
        
        # Save conversation to vector database with emotional state
        if hasattr(self, 'last_user_message') and self.last_user_message:
            try:
                # Get current emotional levels (using lowercase keys)
                fear_level = self.emotion_scores.get('fear', 0.0)
                happiness_level = self.emotion_scores.get('happiness', 0.0) 
                sadness_level = self.emotion_scores.get('sadness', 0.0)
                anger_level = self.emotion_scores.get('anger', 0.0)
                
                print(f"Saving conversation with emotion levels: fear={fear_level}, happiness={happiness_level}, sadness={sadness_level}, anger={anger_level}")
                
                # Get stimulus keys with their actual values from current stimulus
                stimulus_keys = dict(self.current_stimulus) if self.current_stimulus else {}
                
                # Save to vector database
                self.emotional_vector_db.add_conversation(
                    user_character_name=self.user_character_name,
                    ai_character_name=self.character_data['name'],
                    user_message=self.last_user_message,
                    ai_response=cleaned_response,
                    fear_level=fear_level,
                    happiness_level=happiness_level,
                    sadness_level=sadness_level,
                    anger_level=anger_level,
                    stimulus_keys=stimulus_keys
                )
                
                # Clear the stored message
                self.last_user_message = None
                
            except Exception as e:
                print(f"Error saving conversation to vector database: {e}")
        
        # Scroll to bottom
        self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum()
        )
    
    def clean_response_of_function_calls(self, response):
        """Remove function call syntax from the response for cleaner display"""
        # Remove function calls but keep the surrounding text
        function_pattern = r'AI_Factor_Decide\s*\([^)]+\)'
        cleaned = re.sub(function_pattern, '', response, flags=re.IGNORECASE)
        
        # Clean up any extra whitespace or punctuation left behind
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single space
        cleaned = re.sub(r'\s*,\s*,', ',', cleaned)  # Double commas
        cleaned = re.sub(r'^\s*,\s*', '', cleaned)  # Leading comma
        cleaned = re.sub(r'\s*,\s*$', '', cleaned)  # Trailing comma
        
        return cleaned.strip()
    
    def format_function_call_summary(self, executed_functions):
        """Format a summary of executed function calls for display"""
        if not executed_functions:
            return ""
        
        summary_parts = []
        for func_call in executed_functions:
            if func_call['result']['success']:
                factor = func_call['parameters']['factor']
                action = func_call['parameters']['action']
                change = func_call['result']['change']
                new_value = func_call['result']['new_value']
                
                summary_parts.append(f"{factor}: {action}d by {abs(change):.2f} (now {new_value:.2f})")
            else:
                summary_parts.append(f"Failed: {func_call['result']['error']}")
        
        return f"[Emotional adjustments: {', '.join(summary_parts)}]"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Synthetic Emotion Engine")
    app.setApplicationVersion("6.0")
    app.setOrganizationName("SEE Project")
    
    # Set application icon for taskbar and system tray
    try:
        app.setWindowIcon(QIcon(get_app_path("icon.ico")))
    except:
        try:
            app.setWindowIcon(QIcon(get_app_path("icon.png")))
        except:
            print("Warning: Could not load application icon for taskbar")
    
    window = SyntheticEmotionsApp()
    window.show()
    sys.exit(app.exec_())