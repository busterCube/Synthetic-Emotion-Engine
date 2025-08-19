# Synthetic Emotion Engine

A sophisticated AI character interaction system with dynamic emotional modeling, vector-based memory, and text-to-speech capabilities.

## Overview

The Synthetic Emotion Engine (SEE) is an advanced conversational AI system that creates emotionally intelligent characters with persistent memory and dynamic personality traits. Unlike traditional chatbots, SEE characters have complex emotional profiles that evolve in real-time based on conversation context, creating more natural and engaging interactions.

## Key Features

### 🧠 **Dynamic Emotional Modeling**
- **12 Emotional Factors**: Anxiety, Self-Preservation, Concern for Others, Social Connection, Achievement, Sensory Pleasure, Loss, Disappointment, Loneliness, Isolation, Pain, and Frustration
- **Real-time Emotional States**: Characters dynamically adjust their emotional state during conversations using `AI_Factor_Decide()` function calls
- **4 Core Emotions**: Fear, Happiness, Sadness, and Anger with configurable factor weights
- **Visual Emotion Tracking**: Real-time sliders and displays showing current emotional state

### 💾 **Vector-based Memory System**
- **Emotional Context Storage**: Conversations stored with complete emotional state data
- **Character-specific Memory**: Prioritized recall based on user character relationships
- **Semantic Search**: Advanced vector similarity search using sentence transformers
- **Memory Continuity**: Characters reference past emotional states and relationship history

### 🎭 **Character System**
- **JSON-based Character Definitions**: Comprehensive character files with backstory, personality, and emotional profiles
- **Custom Emotional Profiles**: Each character has unique sensitivity to different emotional stimuli
- **Relationship Awareness**: Characters remember and reference their relationship with specific users
- **Character Database Management**: Full CRUD operations for conversation history

### 🔊 **Text-to-Speech Integration**
- **Kokoro TTS Support**: High-quality neural text-to-speech with multiple voice models
- **Character Voice Matching**: Each character can have a specific voice defined in their JSON file
- **Toggle Control**: Easy on/off switch for voice output
- **Clean Speech Processing**: Automatic removal of formatting and action text for natural speech

### 🔗 **LLM Integration**
- **Ollama Compatibility**: Seamless integration with local Ollama models
- **Model Selection**: Support for multiple LLM models with easy switching
- **Advanced Prompting**: Context-aware prompts with emotional state and memory integration

## Prerequisites

### Required Dependencies
```bash
pip install PyQt5 requests numpy sentence-transformers sqlite-vec
```

### Optional Dependencies (for TTS)
```bash
pip install pyaudio kokoro
```

### External Requirements
- **Ollama**: Local LLM server ([Download Ollama](https://ollama.ai/))
- **LLM Models**: Compatible models like Llama 3, Mistral, or other instruction-tuned models

## Installation

1. **Clone or download** the Synthetic Emotion Engine files
2. **Install Python dependencies**:
   ```bash
   pip install PyQt5 requests numpy sentence-transformers sqlite-vec
   ```
3. **Install Ollama** and pull a compatible model:
   ```bash
   ollama pull llama3
   ```
4. **Optional: Install TTS dependencies** for voice output:
   ```bash
   pip install pyaudio kokoro
   ```

## Quick Start

1. **Launch the Application**:
   ```bash
   python Synthetic_Emotion_Engine_v6.py
   ```

2. **Load a Character**:
   - Click "Load Character" and select a JSON character file
   - Sample character: `Persona/Lysandra Vale_SEE.json`

3. **Select an LLM Model**:
   - Choose from available Ollama models in the dropdown
   - Click "Refresh Models" if needed

4. **Start Chatting**:
   - Type messages in the input field
   - Watch emotional states change in real-time
   - Enable TTS for voice responses (if available)

## Usage Guide

### Character Creation

Characters are defined in JSON files with the following structure:

```json
{
  "name": "Character Name",
  "voice": "af_nova",
  "backstory": "Character background...",
  "personality": "Personality traits...",
  "persona": "First-person identity...",
  "character_instructions": "LLM instructions...",
  "emotional_profile": {
    "Anxiety_Factor": 0.05,
    "SocialConnection_Factor": 0.07,
    // ... more factors
  },
  "base_emotion_weights": {
    "fear": {
      "Anxiety_Factor": 0.4,
      "SelfPreservation_Factor": 0.3,
      "ConcernForOthers_Factor": 0.3
    }
    // ... more emotions
  }
}
```

📖 **See `Character_JSON_Guide.md` for complete character creation documentation.**

### Interface Overview

#### Main Chat Tab
- **Emotion Sliders**: Manually adjust emotional factors (for testing)
- **Chat History**: Conversation display with emotional context
- **Quick Emotion Buttons**: Instant emotion state changes
- **Real-time Emotion Display**: Current fear, happiness, sadness, and anger levels

#### Vector Database Tab
- **Conversation History**: View all stored conversations with emotional data
- **Character Filtering**: Filter conversations by AI character
- **Stimulus Values**: See actual emotional factor values for each conversation
- **Database Management**: Delete conversations or clear entire database

#### Top Controls
- **Character Loading**: Load new character JSON files
- **TTS Toggle**: Enable/disable text-to-speech
- **Model Selection**: Choose Ollama LLM model
- **Model Refresh**: Update available model list

### Emotional System

Characters use a sophisticated emotional model:

1. **Emotional Factors** (0.0-1.0): Sensitivity to different stimuli
2. **Stimulus Values** (0.0-1.0): Current activation levels
3. **Emotion Weights**: How factors contribute to core emotions
4. **Dynamic Updates**: Real-time changes via `AI_Factor_Decide()` calls

Example function call:
```
AI_Factor_Decide("Anxiety_Factor", "increase", 0.3)
```

### Memory System

The vector database stores:
- **Conversation Content**: User messages and AI responses
- **Emotional Context**: Complete emotional state at time of conversation
- **Character Relationships**: User character names for personalized recall
- **Stimulus Data**: Exact emotional factor values

Memory retrieval prioritizes:
1. **Same User Character**: Conversations with the same user are weighted higher
2. **Semantic Similarity**: Vector similarity matching for relevant context
3. **Emotional Continuity**: Past emotional states inform current responses

## Compatible LLM Models

### Recommended Models

#### **Best Performance:**
- **Llama 3 (8B/70B)**: Excellent instruction following and creative responses
- **Mistral 7B**: Good balance of speed and quality
- **Qwen 2.5**: Strong multilingual and reasoning capabilities

#### **Character Roleplay Optimized:**
- **Llama 3.1**: Enhanced creative writing and character consistency
- **Neural Chat**: Specialized for conversational AI
- **Vicuna**: Fine-tuned for helpful and engaging conversations

#### **Minimum Requirements:**
- **4GB+ Models**: For basic functionality
- **Instruction-tuned**: Models trained for following complex prompts
- **Context Length**: 4K+ tokens recommended for full memory integration

### Model Configuration Tips

1. **Temperature**: 0.7-0.9 for creative character responses
2. **Top-P**: 0.8-0.95 for balanced randomness
3. **Context Window**: Larger windows enable better memory utilization
4. **System Prompts**: SEE handles complex system prompting automatically

## Troubleshooting

### Common Issues

#### **"No models found" Error**
```
Solution:
1. Ensure Ollama is running: `ollama serve`
2. Pull a model: `ollama pull llama3`
3. Check Ollama API at http://localhost:11434
4. Click "Refresh Models" in the application
```

#### **TTS Not Available**
```
Solution:
1. Install dependencies: `pip install pyaudio kokoro`
2. Check system audio output
3. Verify character JSON has valid "voice" field
4. Restart application after installing TTS dependencies
```

#### **Character Loading Errors**
```
Solution:
1. Validate JSON syntax (use JSON validator)
2. Ensure all required fields are present
3. Check emotional_profile values are 0.0-1.0
4. Verify base_emotion_weights sum to 1.0 for each emotion
```

#### **Slow Response Times**
```
Solution:
1. Use smaller LLM models (7B instead of 70B)
2. Reduce conversation history in vector database
3. Ensure sufficient RAM (8GB+ recommended)
4. Close other GPU-intensive applications
```

#### **Memory/Vector Search Issues**
```
Solution:
1. Check EmotionalVectorDB.db file permissions
2. Reinstall sentence-transformers: `pip install --upgrade sentence-transformers`
3. Clear vector database if corrupted (Vector Database tab > Clear All Data)
4. Restart application to reinitialize embedding model
```

#### **UI Responsiveness**
```
Solution:
1. Disable TTS if causing freezes
2. Use smaller embedding models
3. Reduce max conversation history
4. Check system resources (CPU/RAM usage)
```

### Debug Information

The application provides extensive debug output:
- **Console Logs**: Emotional calculations and vector search results
- **Memory Context**: Shows retrieved conversation memories
- **TTS Status**: Voice loading and speech generation logs
- **Function Calls**: AI emotional adjustments in real-time

### Performance Tips

1. **Hardware Requirements**:
   - **Minimum**: 8GB RAM, integrated graphics
   - **Recommended**: 16GB+ RAM, dedicated GPU for larger models
   - **Storage**: 10GB+ free space for models and databases

2. **Optimization**:
   - Use quantized models (Q4_K_M, Q5_K_M) for better performance
   - Limit vector database size (delete old conversations)
   - Close unused applications when running large models

## Advanced Features

### Function Calling System
Characters can dynamically modify their emotional state using built-in function calls:
- **Syntax**: `AI_Factor_Decide(factor_name, action, value)`
- **Real-time Processing**: Immediate emotional state updates
- **Context Awareness**: Function calls based on conversation events

### Custom Character Development
- **Emotional Profiling**: Create unique personality types
- **Relationship Dynamics**: Configure character-specific interaction patterns  
- **Voice Matching**: Pair characters with appropriate TTS voices
- **Behavioral Consistency**: Maintain character traits across conversations

### Database Management
- **Export/Import**: Backup conversation data
- **Analytics**: Track emotional patterns over time
- **Filtering**: Search conversations by emotion, character, or timeframe
- **Privacy**: Local storage with no external data transmission

## File Structure

```
Synthetic_Emotion_Engine/
├── Synthetic_Emotion_Engine_v6.py    # Main application
├── Character_JSON_Guide.md           # Character creation guide
├── Persona/                          # Character JSON files
│   └── Lysandra Vale_SEE.json       # Example character
├── EmotionalVectorDB.db              # Conversation database
├── icon.png / icon.ico              # Application icons
└── README.md                        # This file
```

## Contributing

The Synthetic Emotion Engine is designed for extensibility:
- **Character Templates**: Create and share character archetypes
- **Emotional Models**: Develop new factor systems
- **UI Enhancements**: Improve visualization and interaction
- **Integration**: Add support for new LLM providers or TTS systems

## License

This project is provided as-is for educational and research purposes. Dependencies may have their own licensing terms.

## Support

For issues, questions, or character development assistance:
1. Check the troubleshooting section above
2. Review the Character JSON Guide for character creation
3. Examine console logs for debug information
4. Test with default character and model configurations

---

**Synthetic Emotion Engine** - Creating emotionally intelligent AI characters with persistent memory and dynamic personalities.

