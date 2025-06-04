"""
Generative Image Feature Extraction System
Uses minimal models to extract comprehensive image features including semantics, 
style, scene details, and emotional tone through generative approaches.
"""

import torch
import numpy as np
from PIL import Image
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    pipeline
)
from typing import Set, List, Dict, Any
import requests
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

model_name = "../models/blip2-2.7b"
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
class GenerativeImageAnalyzer:
    """
    Main class that combines multiple specialized analyzers for comprehensive
    image feature extraction using generative approaches.
    """
    
    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.style_analyzer = StyleAnalyzer()
        self.scene_analyzer = SceneAnalyzer()
        self.emotional_analyzer = EmotionalAnalyzer()
        
    def analyze_image(self, image_path: str) -> Dict[str, Set[str]]:
        """
        Analyze image comprehensively across all dimensions.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with analysis results from each analyzer
        """
        results = {}
        
        try:
            print("Generating image features for semantics...")
            results['semantics'] = self.semantic_analyzer.analyze(image_path)
            print("Generating image features for style...")
            results['style'] = self.style_analyzer.analyze(image_path)
            print("Generating image features for scene...")
            results['scene'] = self.scene_analyzer.analyze(image_path)
            print("Generating image features for emotion...")
            results['emotion'] = self.emotional_analyzer.analyze(image_path)
            
            # Combine unique features
            all_features = set()
            for feature_set in results.values():
                all_features.update(feature_set)
            results['combined'] = all_features

            # convert sets to lists
            for key, value in results.items():
                results[key] = list(value)
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            results = {key: set() for key in ['semantics', 'style', 'scene', 'emotion', 'combined']}
            
        return results


class SemanticAnalyzer:
    """
    Extracts semantic content using BLIP-2 generative captioning with multiple prompts
    to cover objects, actions, relationships, and abstract concepts.
    """
    
    def __init__(self):
        print("Loading BLIP-2 model for semantic analysis...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use smaller BLIP-2 model for efficiency
        self.processor = processor
        self.model = model
        # Prompts to extract different semantic aspects
        self.semantic_prompts = [
            "Describe what you see in this image in detail:",
            "What objects and people are in this image?",
            "What actions or activities are happening?",
            "What is the main subject doing?",
            "Describe the relationships between elements:",
            "What story does this image tell?",
            "List the key visual elements:",
            "What concepts does this image represent?"
        ]
        
    def analyze(self, image_path: str) -> Set[str]:
        """Extract semantic features through generative captioning."""
        try:
            # Load and preprocess image
            if image_path.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            
            semantic_features = set()
            
            # Generate captions with different prompts
            for prompt in self.semantic_prompts:
                try:
                    inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            num_beams=3,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9
                        )
                    
                    caption = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0].strip()
                    
                    # Extract meaningful words from caption
                    features = self._extract_features_from_text(caption)
                    semantic_features.update(features)
                    
                except Exception as e:
                    print(f"Error with prompt '{prompt}': {e}")
                    continue
            
            return semantic_features
            
        except Exception as e:
            print(f"Error in semantic analysis: {e}")
            return set()
    
    def _extract_features_from_text(self, text: str) -> Set[str]:
        """Extract meaningful features from generated text."""
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter meaningful words (nouns, adjectives, verbs)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
            'those', 'there', 'here', 'where', 'when', 'how', 'what', 'who'
        }
        
        meaningful_words = {
            word for word in words 
            if len(word) > 2 and word not in stop_words
        }
        
        return meaningful_words


class StyleAnalyzer:
    """
    Analyzes artistic style, color palette, composition, and visual aesthetics
    using prompted BLIP-2 generation focused on artistic elements.
    """
    
    def __init__(self):
        print("Loading style analysis components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Reuse BLIP-2 model for style analysis
        self.processor = processor
        self.model = model
        
        # Style-focused prompts
        self.style_prompts = [
            "Describe the artistic style of this image:",
            "What colors dominate this image?",
            "Describe the lighting and mood:",
            "What is the composition style?",
            "Is this realistic, abstract, or stylized?",
            "Describe the visual texture and technique:",
            "What art movement does this resemble?",
            "Describe the visual atmosphere:"
        ]
        
        # Predefined style vocabularies for enhancement
        self.style_keywords = {
            'realism', 'abstract', 'impressionist', 'modern', 'contemporary',
            'vintage', 'retro', 'minimalist', 'dramatic', 'soft', 'harsh',
            'warm', 'cool', 'bright', 'dark', 'saturated', 'muted',
            'colorful', 'monochrome', 'vibrant', 'subtle', 'bold', 'delicate'
        }
        
    def analyze(self, image_path: str) -> Set[str]:
        """Extract style and aesthetic features."""
        try:
            # Load image
            if image_path.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            
            style_features = set()
            
            # Generate style descriptions
            for prompt in self.style_prompts:
                try:
                    inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=40,
                            num_beams=3,
                            temperature=0.8,
                            do_sample=True
                        )
                    
                    description = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0].strip()
                    
                    features = self._extract_style_features(description)
                    style_features.update(features)
                    
                except Exception as e:
                    print(f"Error with style prompt: {e}")
                    continue
            
            # Add basic color analysis
            color_features = self._analyze_colors(image)
            style_features.update(color_features)
            
            return style_features
            
        except Exception as e:
            print(f"Error in style analysis: {e}")
            return set()
    
    def _extract_style_features(self, text: str) -> Set[str]:
        """Extract style-related features from generated text."""
        text_lower = text.lower()
        features = set()
        
        # Extract style keywords
        words = re.findall(r'\b\w+\b', text_lower)
        for word in words:
            if word in self.style_keywords or len(word) > 3:
                if word not in {'this', 'that', 'with', 'have', 'been', 'very'}:
                    features.add(word)
        
        return features
    
    def _analyze_colors(self, image: Image.Image) -> Set[str]:
        """Analyze dominant colors in the image."""
        # Convert to numpy array and downsample for efficiency
        img_array = np.array(image.resize((100, 100)))
        
        # Calculate color statistics
        mean_rgb = np.mean(img_array, axis=(0, 1))
        
        color_features = set()
        
        # Determine dominant color tendencies
        r, g, b = mean_rgb
        
        if r > 150 and g < 100 and b < 100:
            color_features.add('red-dominant')
        elif g > 150 and r < 100 and b < 100:
            color_features.add('green-dominant')
        elif b > 150 and r < 100 and g < 100:
            color_features.add('blue-dominant')
        elif r > 150 and g > 150 and b < 100:
            color_features.add('yellow-tones')
        elif r > 150 and b > 150 and g < 100:
            color_features.add('purple-tones')
        elif g > 150 and b > 150 and r < 100:
            color_features.add('cyan-tones')
        
        # Brightness analysis
        brightness = np.mean(mean_rgb)
        if brightness > 180:
            color_features.add('bright')
        elif brightness < 80:
            color_features.add('dark')
        else:
            color_features.add('medium-brightness')
        
        # Saturation estimation
        max_val = np.max(mean_rgb)
        min_val = np.min(mean_rgb)
        saturation = (max_val - min_val) / max_val if max_val > 0 else 0
        
        if saturation > 0.6:
            color_features.add('saturated')
        elif saturation < 0.2:
            color_features.add('desaturated')
        
        return color_features


class SceneAnalyzer:
    """
    Analyzes scene context, environment, setting, and spatial relationships
    using environment-focused generative prompts.
    """
    
    def __init__(self):
        print("Loading scene analysis components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Reuse BLIP-2 for scene analysis
        self.processor = processor
        self.model = model
        
        # Scene-focused prompts
        self.scene_prompts = [
            "Where is this photo taken?",
            "Describe the setting and environment:",
            "Is this indoors or outdoors?",
            "What type of location is this?",
            "Describe the background and surroundings:",
            "What time of day does this appear to be?",
            "What season or weather is shown?",
            "Describe the spatial layout:"
        ]
        
    def analyze(self, image_path: str) -> Set[str]:
        """Extract scene and environmental features."""
        try:
            # Load image
            if image_path.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            
            scene_features = set()
            
            # Generate scene descriptions
            for prompt in self.scene_prompts:
                try:
                    inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=35,
                            num_beams=3,
                            temperature=0.7
                        )
                    
                    description = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0].strip()
                    
                    features = self._extract_scene_features(description)
                    scene_features.update(features)
                    
                except Exception as e:
                    print(f"Error with scene prompt: {e}")
                    continue
            
            return scene_features
            
        except Exception as e:
            print(f"Error in scene analysis: {e}")
            return set()
    
    def _extract_scene_features(self, text: str) -> Set[str]:
        """Extract scene-related features from generated text."""
        text_lower = text.lower()
        
        # Scene vocabulary
        scene_keywords = {
            'indoor', 'outdoor', 'inside', 'outside', 'interior', 'exterior',
            'kitchen', 'bedroom', 'living', 'office', 'restaurant', 'cafe',
            'street', 'park', 'garden', 'forest', 'beach', 'mountain',
            'city', 'urban', 'rural', 'suburban', 'natural', 'artificial',
            'day', 'night', 'morning', 'afternoon', 'evening', 'sunset',
            'sunny', 'cloudy', 'rainy', 'snowy', 'winter', 'summer',
            'spring', 'autumn', 'fall', 'bright', 'dim', 'lit'
        }
        
        features = set()
        words = re.findall(r'\b\w+\b', text_lower)
        
        for word in words:
            if word in scene_keywords:
                features.add(word)
            elif len(word) > 4 and word.endswith('ing'):
                features.add(word)  # Capture activity words
        
        return features


class EmotionalAnalyzer:
    """
    Analyzes emotional tone, mood, and psychological impact using 
    emotion-focused generative prompts and sentiment analysis.
    """
    
    def __init__(self):
        print("Loading emotional analysis components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Reuse BLIP-2 for emotional analysis
        self.processor = processor
        self.model = model
        
        # Load sentiment analysis pipeline for additional emotion detection
        try:
            self.sentiment_analyzer = pipeline(
                "text-classification", 
                model="cardiffnlp/twitter-roberta-base-emotion",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            self.sentiment_analyzer = None
        
        # Emotion-focused prompts
        self.emotion_prompts = [
            "What mood does this image convey?",
            "How does this image make you feel?",
            "Describe the emotional atmosphere:",
            "Is this image happy, sad, peaceful, or energetic?",
            "What emotions do the subjects appear to feel?",
            "Describe the psychological impact:",
            "What feelings does this evoke?"
        ]
        
    def analyze(self, image_path: str) -> Set[str]:
        """Extract emotional and mood features."""
        try:
            # Load image
            if image_path.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            
            emotion_features = set()
            all_descriptions = []
            
            # Generate emotion descriptions
            for prompt in self.emotion_prompts:
                try:
                    inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=30,
                            num_beams=3,
                            temperature=0.8
                        )
                    
                    description = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0].strip()
                    
                    all_descriptions.append(description)
                    features = self._extract_emotion_features(description)
                    emotion_features.update(features)
                    
                except Exception as e:
                    print(f"Error with emotion prompt: {e}")
                    continue
            
            # Additional sentiment analysis if available
            if self.sentiment_analyzer and all_descriptions:
                combined_text = " ".join(all_descriptions)
                try:
                    sentiment_results = self.sentiment_analyzer(combined_text)
                    for result in sentiment_results:
                        emotion_features.add(result['label'].lower())
                except:
                    pass
            
            return emotion_features
            
        except Exception as e:
            print(f"Error in emotional analysis: {e}")
            return set()
    
    def _extract_emotion_features(self, text: str) -> Set[str]:
        """Extract emotion-related features from generated text."""
        text_lower = text.lower()
        
        # Emotion vocabulary
        emotion_keywords = {
            'happy', 'sad', 'joy', 'peaceful', 'calm', 'serene', 'tranquil',
            'energetic', 'excited', 'cheerful', 'melancholy', 'nostalgic',
            'romantic', 'dreamy', 'mysterious', 'dramatic', 'intense',
            'playful', 'serious', 'contemplative', 'uplifting', 'soothing',
            'vibrant', 'dynamic', 'static', 'emotional', 'neutral',
            'warm', 'cold', 'inviting', 'lonely', 'crowded', 'intimate',
            'grand', 'cozy', 'comfortable', 'tense', 'relaxed'
        }
        
        features = set()
        words = re.findall(r'\b\w+\b', text_lower)
        
        for word in words:
            if word in emotion_keywords:
                features.add(word)
        
        return features


# Usage example and utility functions
def demo_analysis(image_path: str):
    """
    Demonstrate the complete analysis pipeline.
    """
    print(f"Analyzing image: {image_path}")
    print("=" * 50)
    
    analyzer = GenerativeImageAnalyzer()
    results = analyzer.analyze_image(image_path)
    
    for category, features in results.items():
        if category != 'combined':
            print(f"\n{category.upper()} FEATURES:")
            print(", ".join(sorted(features)) if features else "No features detected")
    
    print(f"\nTOTAL UNIQUE FEATURES: {len(results['combined'])}")
    return results


if __name__ == "__main__":
    # Example usage
    print("Generative Image Feature Extraction System")
    print("==========================================")
    
    # You can test with a sample image
    sample_image = "../docs/astronaut_rides_horse.png"
    
    try:
        results = demo_analysis(sample_image)
    except Exception as e:
        print(f"Demo failed: {e}")
        print("\nTo use this system:")
        print("1. Install requirements: pip install torch transformers pillow requests numpy")
        print("2. Call: analyzer = GenerativeImageAnalyzer()")
        print("3. Use: results = analyzer.analyze_image('path/to/image.jpg')")