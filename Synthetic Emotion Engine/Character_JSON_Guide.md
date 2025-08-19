# Synthetic Emotion Engine Character JSON Guide

## Overview

The Synthetic Emotion Engine uses JSON files to define characters with complex emotional profiles and personalities. Each character file contains multiple layers of configuration that control how the AI responds emotionally to different stimuli and situations.

## File Structure

A character JSON file consists of seven main sections:

1. **Basic Character Information**
2. **Character Definition**
3. **Emotional Profile**
4. **Base Emotion Weights**

---

## 1. Basic Character Information

### `name` (string)
```json
"name": "Lysandra Vale"
```
- **Purpose**: The character's full name displayed in the UI
- **Usage**: Used for conversation logging, database storage, and user interface display
- **Impact**: This name appears in conversation memories and vector database entries

### `voice` (string) 
```json
"voice": "af_nova"
```
- **Purpose**: Specifies which Kokoro TTS voice model to use for speech synthesis
- **Current Status**: **NOT IMPLEMENTED** in this version of the Synthetic Emotion Engine
- **Future Use**: Will control text-to-speech voice characteristics when audio generation is added
- **Valid Values**: Any Kokoro TTS voice identifier (e.g., "af_nova", "am_adam", "bf_emma", etc.)

---

## 2. Character Definition

### `backstory` (string)
```json
"backstory": "Lysandra was raised in the twilight groves of Hollowfern, where the veil between worlds is thin..."
```
- **Purpose**: Provides rich background context for the character
- **Usage**: Not directly sent to the LLM but informs character development
- **Impact**: Helps maintain consistency in character responses and world-building
- **Best Practices**: 
  - Keep to 2-3 sentences
  - Include key formative experiences
  - Establish the character's world/setting

### `personality` (string)
```json
"personality": "Serene and enigmatic, Lysandra speaks in poetic metaphors and often pauses to listen to the wind..."
```
- **Purpose**: Defines core personality traits and behavioral patterns
- **Usage**: Influences how emotional states manifest in responses
- **Impact**: Shapes tone, speech patterns, and decision-making tendencies
- **Best Practices**:
  - Focus on observable behaviors
  - Include communication style
  - Mention conflict resolution preferences

### `persona` (string)
```json
"persona": "I am Lysandra Vale, voice of the glade and sentinel of the old ways..."
```
- **Purpose**: The character's self-perception and identity statement
- **Usage**: Included in LLM prompts to reinforce character identity
- **Impact**: Strengthens character consistency and voice
- **Best Practices**:
  - Write in first person
  - Include core values and purpose
  - Reflect the character's worldview

### `character_instructions` (string)
```json
"character_instructions": "CRITICAL: You are embodying a character. NEVER mention JSON, data fields..."
```
- **Purpose**: Technical instructions for the LLM on how to portray the character
- **Usage**: Directly included in every prompt sent to the LLM
- **Impact**: Prevents meta-references and maintains immersion
- **Standard Template**: Use the provided template to maintain consistency across characters

---

## 3. Emotional Profile

The `emotional_profile` section defines how sensitive the character is to different emotional stimuli. Each factor represents a baseline sensitivity level (0.0 to 1.0).

```json
"emotional_profile": {
    "Anxiety_Factor": 0.05,
    "SelfPreservation_Factor": 0.03,
    "ConcernForOthers_Factor": 0.04,
    "SocialConnection_Factor": 0.07,
    "Achievement_Factor": 0.06,
    "SensoryPleasure_Factor": 0.08,
    "Loss_Factor": 0.04,
    "Disappointment_Factor": 0.05,
    "Loneliness_Factor": 0.03,
    "Isolation_Factor": 0.02,
    "Pain_Factor": 0.06,
    "Frustration_Factor": 0.08
}
```

### Factor Definitions:

#### Fear-Related Factors:
- **`Anxiety_Factor`** (0.0-1.0): Sensitivity to worry, unease, and anticipation of threats
  - *Low (0.01-0.03)*: Calm, unflappable character
  - *Medium (0.04-0.07)*: Balanced, appropriate caution
  - *High (0.08-1.0)*: Nervous, easily worried character

- **`SelfPreservation_Factor`** (0.0-1.0): Concern for personal safety and survival
  - *Low*: Reckless, self-sacrificing tendencies
  - *High*: Highly cautious, survival-focused

- **`ConcernForOthers_Factor`** (0.0-1.0): Empathy and worry for others' wellbeing
  - *Low*: Self-centered, indifferent to others' pain
  - *High*: Deeply empathetic, easily affected by others' suffering

#### Happiness-Related Factors:
- **`SocialConnection_Factor`** (0.0-1.0): Desire for belonging and social bonds
  - *Low*: Introverted, solitary preferences
  - *High*: Extroverted, relationship-focused

- **`Achievement_Factor`** (0.0-1.0): Pride and drive from accomplishments
  - *Low*: Unmotivated by success, humble
  - *High*: Ambitious, achievement-oriented

- **`SensoryPleasure_Factor`** (0.0-1.0): Enjoyment of physical comfort and beauty
  - *Low*: Ascetic, indifferent to material pleasures
  - *High*: Hedonistic, pleasure-seeking

#### Sadness-Related Factors:
- **`Loss_Factor`** (0.0-1.0): Emotional pain from losing cherished things/people
  - *Low*: Emotionally detached, quick to move on
  - *High*: Deeply affected by loss, may struggle with grief

- **`Disappointment_Factor`** (0.0-1.0): Sensitivity to unmet expectations
  - *Low*: Resilient, adaptable to setbacks
  - *High*: Easily discouraged, sensitive to failure

- **`Loneliness_Factor`** (0.0-1.0): Emotional emptiness from lack of connection
  - *Low*: Comfortable with solitude
  - *High*: Needs constant social interaction

#### Anger-Related Factors:
- **`Isolation_Factor`** (0.0-1.0): Frustration from being cut off from others
  - *Low*: Independent, self-sufficient
  - *High*: Becomes angry when excluded or separated

- **`Pain_Factor`** (0.0-1.0): Reaction to physical or emotional suffering
  - *Low*: High pain tolerance, stoic
  - *High*: Sensitive to pain, easily triggered by suffering

- **`Frustration_Factor`** (0.0-1.0): Agitation from blocked goals or repeated failure
  - *Low*: Patient, persistent
  - *High*: Quick to anger when thwarted

---

## 4. Base Emotion Weights

The `base_emotion_weights` section defines how different sensitivity factors contribute to the four core emotions. Each emotion has associated factors with weights that must sum to 1.0.

```json
"base_emotion_weights": {
    "fear": {
        "Anxiety_Factor": 0.4,
        "SelfPreservation_Factor": 0.3,
        "ConcernForOthers_Factor": 0.3
    },
    "happiness": {
        "SocialConnection_Factor": 0.5,
        "Achievement_Factor": 0.3,
        "SensoryPleasure_Factor": 0.2
    },
    "sadness": {
        "Loss_Factor": 0.5,
        "Disappointment_Factor": 0.3,
        "Loneliness_Factor": 0.2
    },
    "anger": {
        "Isolation_Factor": 0.3,
        "Pain_Factor": 0.4,
        "Frustration_Factor": 0.3
    }
}
```

### Weight Configuration Examples:

#### Fear Emotion:
```json
"fear": {
    "Anxiety_Factor": 0.4,        // 40% of fear comes from general anxiety
    "SelfPreservation_Factor": 0.3,  // 30% from survival instincts
    "ConcernForOthers_Factor": 0.3   // 30% from worry about others
}
```

**Character Variations:**
- **Protective Character**: `ConcernForOthers_Factor: 0.6, Anxiety_Factor: 0.2, SelfPreservation_Factor: 0.2`
- **Survival-Focused**: `SelfPreservation_Factor: 0.7, Anxiety_Factor: 0.2, ConcernForOthers_Factor: 0.1`
- **Anxious Character**: `Anxiety_Factor: 0.7, SelfPreservation_Factor: 0.2, ConcernForOthers_Factor: 0.1`

#### Happiness Emotion:
```json
"happiness": {
    "SocialConnection_Factor": 0.5,  // 50% from relationships and belonging
    "Achievement_Factor": 0.3,       // 30% from accomplishments
    "SensoryPleasure_Factor": 0.2    // 20% from physical pleasures
}
```

**Character Variations:**
- **Social Butterfly**: `SocialConnection_Factor: 0.8, Achievement_Factor: 0.1, SensoryPleasure_Factor: 0.1`
- **Ambitious Go-Getter**: `Achievement_Factor: 0.6, SocialConnection_Factor: 0.3, SensoryPleasure_Factor: 0.1`
- **Hedonistic Character**: `SensoryPleasure_Factor: 0.6, SocialConnection_Factor: 0.3, Achievement_Factor: 0.1`

---

## How the System Works

### Emotion Calculation Formula:
```
Emotion Score = Σ(Factor_Weight × Factor_Sensitivity × Current_Stimulus_Value)
```

### Example Calculation:
If a character with the above profile experiences:
- Anxiety stimulus: 0.3
- SelfPreservation stimulus: 0.1
- ConcernForOthers stimulus: 0.0

**Fear Score** = (0.4 × 0.05 × 0.3) + (0.3 × 0.03 × 0.1) + (0.3 × 0.04 × 0.0)
             = 0.006 + 0.0009 + 0.0
             = 0.0069

### Real-Time Adjustment:
The AI can modify stimulus values during conversation using:
```
AI_Factor_Decide("Anxiety_Factor", "increase", 0.3)
```

This dynamically adjusts the character's emotional state based on story events.

---

## Character Design Guidelines

### 1. Balanced Characters
- Avoid extreme values (all 0.01 or all 1.0)
- Most factors should range 0.02-0.10 for realistic responses
- Use higher values (0.15+) sparingly for defining traits

### 2. Personality Consistency
- High `SocialConnection_Factor` + Low `Isolation_Factor` = Extroverted
- High `Achievement_Factor` + Low `Disappointment_Factor` = Confident
- High `ConcernForOthers_Factor` + Low `SelfPreservation_Factor` = Self-sacrificing

### 3. Weight Distribution
- Ensure each emotion's weights sum to 1.0
- Primary emotion drivers should have weights 0.4-0.6
- Secondary drivers: 0.2-0.4
- Minor influences: 0.1-0.2

### 4. Testing and Refinement
- Start with moderate values
- Test character responses to various stimuli
- Adjust factors based on desired personality traits
- Use the Vector Database tab to monitor emotional patterns

---

## Example Character Archetypes

### The Protector
```json
"emotional_profile": {
    "ConcernForOthers_Factor": 0.12,
    "SelfPreservation_Factor": 0.02,
    "SocialConnection_Factor": 0.08,
    "Anxiety_Factor": 0.06
}
"base_emotion_weights": {
    "fear": {
        "ConcernForOthers_Factor": 0.7,
        "Anxiety_Factor": 0.2,
        "SelfPreservation_Factor": 0.1
    }
}
```

### The Scholar
```json
"emotional_profile": {
    "Achievement_Factor": 0.10,
    "Frustration_Factor": 0.09,
    "SocialConnection_Factor": 0.03,
    "Disappointment_Factor": 0.08
}
"base_emotion_weights": {
    "happiness": {
        "Achievement_Factor": 0.8,
        "SensoryPleasure_Factor": 0.1,
        "SocialConnection_Factor": 0.1
    }
}
```

### The Warrior
```json
"emotional_profile": {
    "Pain_Factor": 0.03,
    "Frustration_Factor": 0.11,
    "SelfPreservation_Factor": 0.08,
    "Achievement_Factor": 0.09
}
"base_emotion_weights": {
    "anger": {
        "Frustration_Factor": 0.5,
        "Pain_Factor": 0.3,
        "Isolation_Factor": 0.2
    }
}
```

---

## Integration with Vector Database

The emotional factors and their values are:
1. **Stored** in the conversation database with each interaction
2. **Retrieved** during memory searches to provide emotional context
3. **Displayed** in the Vector Database tab showing active stimuli values
4. **Used** to inform future responses based on emotional history

This creates a comprehensive emotional memory system that allows characters to:
- Reference past emotional states
- Show emotional growth or regression
- Maintain personality consistency across conversations
- Develop deeper relationships based on shared emotional experiences

---

## File Naming Convention

Save character files as: `[Character Name]_SEE.json`

Example: `Lysandra Vale_SEE.json`

The `_SEE` suffix identifies files as Synthetic Emotion Engine character definitions.
