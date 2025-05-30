from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import librosa
import torch
import tensorflow as tf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import tempfile
import os
import logging
from werkzeug.utils import secure_filename
import warnings
from collections import Counter
import math
import time
import json
from datetime import datetime
import subprocess
import sys

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

class MultimodalEmotionDetector:
    
    def __init__(self):
        """Initialize all the emotion detection models"""
        print("Loading emotion detection models...")
        
        # Text emotion model (Hugging Face)
        self.text_model_name = "j-hartmann/emotion-english-distilroberta-base"
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(self.text_model_name)
            self.text_classifier = pipeline(
                "text-classification",
                model=self.text_model,
                tokenizer=self.text_tokenizer,
                return_all_scores=True
            )
            print("✅ Text emotion model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading text model: {e}")
            self.text_classifier = None
        
        # Video emotion model (OpenCV + Deep Learning)
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.video_model = self.build_emotion_model()
            print("✅ Video emotion model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading video model: {e}")
            self.face_cascade = None
            self.video_model = None
        
        # Audio emotion features
        self.audio_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Create temp directory if it doesn't exist
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        print("🎉 All models loaded successfully!")
    
    def build_emotion_model(self):
        """Build a CNN model for facial emotion recognition"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(7, activation='softmax')
            ])
            
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            print(f"Error building emotion model: {e}")
            return None
    
    def detect_text_emotion(self, text):
        """Detect emotion from text using Hugging Face transformers"""
        if not text or len(text.strip()) < 3:
            return {"emotion": "neutral", "confidence": 0.5, "all_scores": [], "error": None}
        
        if not self.text_classifier:
            return {"emotion": "neutral", "confidence": 0.5, "all_scores": [], "error": "Text model not loaded"}
        
        try:
            results = self.text_classifier(text)
            
            # Get the highest scoring emotion
            best_result = max(results[0], key=lambda x: x['score'])
            
            # Map emotions to standard set
            emotion_mapping = {
                'joy': 'happy',
                'sadness': 'sad',
                'anger': 'angry',
                'fear': 'fear',
                'surprise': 'surprise',
                'disgust': 'disgust',
                'neutral': 'neutral'
            }
            
            mapped_emotion = emotion_mapping.get(best_result['label'].lower(), best_result['label'].lower())
            
            return {
                "emotion": mapped_emotion,
                "confidence": float(best_result['score']),
                "all_scores": [{"emotion": emotion_mapping.get(r['label'].lower(), r['label'].lower()), 
                               "score": float(r['score'])} for r in results[0]],
                "original_text": text,
                "error": None
            }
        except Exception as e:
            print(f"Error in text emotion detection: {e}")
            return {"emotion": "neutral", "confidence": 0.5, "all_scores": [], "error": str(e)}
    
    def detect_video_emotion(self, video_path):
        """Detect emotion from video using facial expression analysis"""
        if not self.face_cascade:
            return {"emotion": "neutral", "confidence": 0.5, "frames_processed": 0, "error": "Video model not loaded"}
        
        try:
            # Convert webm to mp4 if needed
            converted_path = self.convert_video_format(video_path)
            
            cap = cv2.VideoCapture(converted_path)
            if not cap.isOpened():
                cap = cv2.VideoCapture(video_path)  # Try original file
            
            emotions_detected = []
            frame_count = 0
            faces_found = 0
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process every nth frame based on video length
            skip_frames = max(1, int(fps / 3))  # Process 3 frames per second
            
            while cap.isOpened() and frame_count < min(300, total_frames):  # Limit processing
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % skip_frames == 0:  # Process selected frames
                    emotion_data = self.process_video_frame(frame)
                    if emotion_data:
                        emotions_detected.append(emotion_data['emotion'])
                        faces_found += 1
                
                frame_count += 1
            
            cap.release()
            
            # Clean up converted file
            if converted_path != video_path and os.path.exists(converted_path):
                try:
                    os.remove(converted_path)
                except:
                    pass
            
            if emotions_detected:
                # Calculate emotion distribution
                emotion_counter = Counter(emotions_detected)
                most_common = emotion_counter.most_common(1)[0]
                confidence = (most_common[1] / len(emotions_detected)) * 0.9  # Scale confidence
                
                return {
                    "emotion": most_common[0],
                    "confidence": min(confidence + 0.1, 1.0),  # Add small boost
                    "frames_processed": len(emotions_detected),
                    "faces_found": faces_found,
                    "emotion_distribution": dict(emotion_counter),
                    "video_duration": frame_count / fps if fps > 0 else 0,
                    "error": None
                }
            else:
                return {
                    "emotion": "neutral", 
                    "confidence": 0.6, 
                    "frames_processed": frame_count,
                    "faces_found": faces_found,
                    "error": "No faces detected in video"
                }
                
        except Exception as e:
            print(f"Error in video emotion detection: {e}")
            return {"emotion": "neutral", "confidence": 0.5, "frames_processed": 0, "error": str(e)}
    
    def convert_video_format(self, input_path):
        """Convert video to a more compatible format using ffmpeg if available"""
        try:
            output_path = input_path.replace('.webm', '_converted.mp4')
            
            # Check if ffmpeg is available
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                return input_path  # ffmpeg not available, return original
            
            # Convert video
            cmd = [
                'ffmpeg', '-i', input_path, 
                '-c:v', 'libx264', 
                '-c:a', 'aac', 
                '-y',  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            else:
                return input_path
                
        except Exception as e:
            print(f"Video conversion failed: {e}")
            return input_path
    
    def process_video_frame(self, frame):
        """Process a single video frame for emotion detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
            
            # Process the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract and preprocess face
            face = gray[y:y+h, x:x+w]
            
            if face.size == 0:
                return None
            
            # Resize to standard size
            face_resized = cv2.resize(face, (48, 48))
            
            # Enhance contrast
            face_enhanced = cv2.equalizeHist(face_resized)
            
            # Predict emotion
            emotion = self.predict_face_emotion_advanced(face_enhanced)
            
            return {
                "emotion": emotion["emotion"],
                "confidence": emotion["confidence"],
                "face_size": (w, h)
            }
            
        except Exception as e:
            print(f"Error processing video frame: {e}")
            return None
    
    def predict_face_emotion_advanced(self, face_image):
        """Advanced emotion prediction based on facial features"""
        try:
            # Normalize image
            face_normalized = face_image.astype(np.float32) / 255.0
            
            # Extract features
            features = self.extract_facial_features(face_normalized)
            
            # Rule-based classification with multiple features
            emotion_scores = {
                'happy': 0,
                'sad': 0,
                'angry': 0,
                'fear': 0,
                'surprise': 0,
                'disgust': 0,
                'neutral': 0
            }
            
            # Analyze different facial regions
            mean_intensity = np.mean(face_normalized)
            std_intensity = np.std(face_normalized)
            
            # Eye region analysis (upper 40% of face)
            eye_region = face_normalized[:19, :]
            eye_intensity = np.mean(eye_region)
            
            # Mouth region analysis (lower 30% of face)
            mouth_region = face_normalized[34:, :]
            mouth_intensity = np.mean(mouth_region)
            mouth_std = np.std(mouth_region)
            
            # Gradient analysis for expression lines
            grad_x = cv2.Sobel(face_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            
            # Emotion classification rules
            
            # Happy: Higher mouth region intensity, moderate gradients
            if mouth_intensity > mean_intensity * 1.1 and mouth_std > 0.15:
                emotion_scores['happy'] += 0.7
            
            # Sad: Lower overall intensity, higher eye region intensity
            if mean_intensity < 0.4 and eye_intensity > mouth_intensity:
                emotion_scores['sad'] += 0.6
            
            # Angry: High gradient magnitude, lower eye region
            if avg_gradient > 15 and eye_intensity < mean_intensity * 0.9:
                emotion_scores['angry'] += 0.65
            
            # Fear: High std, moderate gradients
            if std_intensity > 0.2 and avg_gradient > 12:
                emotion_scores['fear'] += 0.6
            
            # Surprise: Very high std, high eye region
            if std_intensity > 0.25 and eye_intensity > mean_intensity * 1.2:
                emotion_scores['surprise'] += 0.7
            
            # Disgust: Specific mouth-eye relationship
            if mouth_intensity < mean_intensity * 0.85 and eye_intensity > mean_intensity:
                emotion_scores['disgust'] += 0.55
            
            # Neutral: Balanced features
            if abs(eye_intensity - mouth_intensity) < 0.05 and std_intensity < 0.18:
                emotion_scores['neutral'] += 0.6
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores.keys(), key=lambda x: emotion_scores[x])
            confidence = emotion_scores[dominant_emotion]
            
            # If no strong emotion detected, default to neutral
            if confidence < 0.4:
                return {"emotion": "neutral", "confidence": 0.6}
            
            return {
                "emotion": dominant_emotion,
                "confidence": min(confidence + 0.2, 1.0)  # Boost confidence
            }
            
        except Exception as e:
            print(f"Error in advanced face emotion prediction: {e}")
            return {"emotion": "neutral", "confidence": 0.5}
    
    def extract_facial_features(self, face_image):
        """Extract various facial features for emotion analysis"""
        features = {}
        
        try:
            # Basic statistical features
            features['mean_intensity'] = np.mean(face_image)
            features['std_intensity'] = np.std(face_image)
            features['min_intensity'] = np.min(face_image)
            features['max_intensity'] = np.max(face_image)
            
            # Texture features using Local Binary Patterns concept
            h, w = face_image.shape
            
            # Divide face into regions
            features['upper_mean'] = np.mean(face_image[:h//3, :])  # Forehead
            features['middle_mean'] = np.mean(face_image[h//3:2*h//3, :])  # Eyes/nose
            features['lower_mean'] = np.mean(face_image[2*h//3:, :])  # Mouth
            
            return features
            
        except Exception as e:
            print(f"Error extracting facial features: {e}")
            return {}
    
    def detect_audio_emotion(self, audio_path):
        """Detect emotion from audio using speech analysis"""
        try:
            # Convert audio format if needed
            converted_path = self.convert_audio_format(audio_path)
            
            # Load audio file
            y, sr = librosa.load(converted_path, duration=30, sr=22050)
            
            if len(y) == 0:
                return {"emotion": "neutral", "confidence": 0.5, "features": {}, "error": "Empty audio"}
            
            # Clean up converted file
            if converted_path != audio_path and os.path.exists(converted_path):
                try:
                    os.remove(converted_path)
                except:
                    pass
            
            # Extract comprehensive audio features
            features = self.extract_comprehensive_audio_features(y, sr)
            
            # Classify emotion using enhanced rules
            emotion_result = self.classify_audio_emotion_advanced(features)
            
            return {
                "emotion": emotion_result["emotion"],
                "confidence": emotion_result["confidence"],
                "features": features,
                "audio_duration": len(y) / sr,
                "sample_rate": sr,
                "error": None
            }
            
        except Exception as e:
            print(f"Error in audio emotion detection: {e}")
            return {"emotion": "neutral", "confidence": 0.5, "features": {}, "error": str(e)}
    
    def convert_audio_format(self, input_path):
        """Convert audio to WAV format using ffmpeg if available"""
        try:
            output_path = input_path.replace('.webm', '_converted.wav')
            
            # Check if ffmpeg is available
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                return input_path
            
            # Convert audio
            cmd = [
                'ffmpeg', '-i', input_path,
                '-ar', '22050',  # Sample rate
                '-ac', '1',      # Mono
                '-y',            # Overwrite
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            else:
                return input_path
                
        except Exception as e:
            print(f"Audio conversion failed: {e}")
            return input_path
    
    def extract_comprehensive_audio_features(self, y, sr):
        """Extract comprehensive features from audio for emotion detection"""
        features = {}
        
        try:
            # Basic features
            features['duration'] = len(y) / sr
            features['mean_amplitude'] = np.mean(np.abs(y))
            features['std_amplitude'] = np.std(y)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # MFCC features (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # Tempo and beat
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = tempo
                features['beat_count'] = len(beats)
            except:
                features['tempo'] = 120
                features['beat_count'] = 0
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # Pitch features
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    features['pitch_mean'] = np.mean(pitch_values)
                    features['pitch_std'] = np.std(pitch_values)
                    features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
                else:
                    features['pitch_mean'] = 0
                    features['pitch_std'] = 0
                    features['pitch_range'] = 0
            except:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
            
            # Spectral contrast
            try:
                contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                features['spectral_contrast_mean'] = np.mean(contrast)
                features['spectral_contrast_std'] = np.std(contrast)
            except:
                features['spectral_contrast_mean'] = 0
                features['spectral_contrast_std'] = 0
            
        except Exception as e:
            print(f"Error extracting comprehensive audio features: {e}")
            # Return default features
            features = {
                'spectral_centroid_mean': 2000,
                'zcr_mean': 0.1,
                'mfcc_0_mean': -200,
                'tempo': 120,
                'rms_mean': 0.1,
                'pitch_mean': 200
            }
        
        return features
    
    def classify_audio_emotion_advanced(self, features):
        """Advanced emotion classification based on comprehensive audio features"""
        try:
            # Initialize emotion scores
            emotion_scores = {
                'happy': 0,
                'sad': 0,
                'angry': 0,
                'fear': 0,
                'surprise': 0,
                'disgust': 0,
                'neutral': 0
            }
            
            # Extract key features with defaults
            spectral_centroid = features.get('spectral_centroid_mean', 2000)
            zcr = features.get('zcr_mean', 0.1)
            mfcc_0 = features.get('mfcc_0_mean', -200)
            tempo = features.get('tempo', 120)
            rms_energy = features.get('rms_mean', 0.1)
            pitch_mean = features.get('pitch_mean', 200)
            pitch_std = features.get('pitch_std', 50)
            spectral_rolloff = features.get('spectral_rolloff_mean', 3000)
            
            # Happy: High energy, higher pitch, faster tempo
            if rms_energy > 0.12 and pitch_mean > 180 and tempo > 110:
                emotion_scores['happy'] += 0.8
            if spectral_centroid > 2200 and zcr > 0.08:
                emotion_scores['happy'] += 0.3
            
            # Sad: Low energy, lower pitch, slower tempo
            if rms_energy < 0.08 and pitch_mean < 150 and tempo < 100:
                emotion_scores['sad'] += 0.7
            if mfcc_0 < -250 and spectral_centroid < 1800:
                emotion_scores['sad'] += 0.4
            
            # Angry: High energy, variable pitch, harsh timbre
            if rms_energy > 0.15 and pitch_std > 60:
                emotion_scores['angry'] += 0.75
            if spectral_centroid > 2500 and zcr > 0.12:
                emotion_scores['angry'] += 0.35
            if spectral_rolloff > 4000:
                emotion_scores['angry'] += 0.2
            
            # Fear: Moderate energy, higher pitch variation, trembling
            if pitch_std > 80 and rms_energy > 0.06 and rms_energy < 0.12:
                emotion_scores['fear'] += 0.65
            if zcr > 0.15 and spectral_centroid > 2000:
                emotion_scores['fear'] += 0.3
            
            # Surprise: Sudden energy changes, higher pitch
            if pitch_mean > 200 and rms_energy > 0.1:
                emotion_scores['surprise'] += 0.6
            if spectral_centroid > 2400 and zcr > 0.1:
                emotion_scores['surprise'] += 0.3
            
            # Disgust: Lower energy, specific spectral characteristics
            if rms_energy < 0.1 and spectral_centroid < 2000:
                emotion_scores['disgust'] += 0.5
            if mfcc_0 < -220 and pitch_mean < 170:
                emotion_scores['disgust'] += 0.3
            
            # Neutral: Balanced characteristics
            if (0.08 <= rms_energy <= 0.12 and 
                150 <= pitch_mean <= 200 and 
                100 <= tempo <= 130):
                emotion_scores['neutral'] += 0.6
            
            # Additional scoring based on MFCC patterns
            mfcc_1_mean = features.get('mfcc_1_mean', 0)
            mfcc_2_mean = features.get('mfcc_2_mean', 0)
            
            # Emotional prosody patterns
            if mfcc_1_mean > 10:  # Often associated with happiness
                emotion_scores['happy'] += 0.2
            elif mfcc_1_mean < -10:  # Often associated with sadness
                emotion_scores['sad'] += 0.2
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores.keys(), key=lambda x: emotion_scores[x])
            max_score = emotion_scores[dominant_emotion]
            
            # Calculate confidence based on score and feature consistency
            confidence = min(max_score, 1.0)
            
            # Boost confidence if multiple indicators align
            if max_score > 0.8:
                confidence = min(confidence + 0.1, 1.0)
            
            # Minimum confidence threshold
            if confidence < 0.4:
                dominant_emotion = 'neutral'
                confidence = 0.6
            
            return {
                "emotion": dominant_emotion,
                "confidence": confidence,
                "emotion_scores": emotion_scores
            }
                
        except Exception as e:
            print(f"Error in advanced audio emotion classification: {e}")
            return {"emotion": "neutral", "confidence": 0.5, "emotion_scores": {}}
    
    def combine_emotions(self, video_result, audio_result, text_result):
        """Combine emotions from all modalities using weighted average and confidence"""
        
        # Define emotion mappings to standardize
        emotion_map = {
            'joy': 'happy', 'happiness': 'happy', 'excited': 'happy',
            'sadness': 'sad', 'sorrow': 'sad',
            'anger': 'angry', 'rage': 'angry',
            'fear': 'fear', 'anxiety': 'fear',
            'surprise': 'surprise', 'shock': 'surprise',
            'disgust': 'disgust', 'contempt': 'disgust',
            'neutral': 'neutral', 'calm': 'neutral'
        }
        
        # Normalize emotions
        def normalize_emotion(emotion):
            return emotion_map.get(emotion.lower(), emotion.lower())
        
        # Collect all emotions with their weights and confidences
        emotions_data = []
        modality_info = {}
        
        # Video emotion (35% weight)
        if video_result and video_result.get('emotion') and not video_result.get('error'):
            confidence = video_result.get('confidence', 0.5)
            # Adjust confidence based on face detection quality
            faces_found = video_result.get('faces_found', 0)
            frames_processed = video_result.get('frames_processed', 0)
            
            if faces_found > 0 and frames_processed > 5:
                confidence_boost = min(0.1, faces_found / frames_processed * 0.2)
                confidence = min(confidence + confidence_boost, 1.0)
            
            emotions_data.append({
                'emotion': normalize_emotion(video_result['emotion']),
                'confidence': confidence,
                'weight': 0.35,
                'modality': 'video'
            })
            modality_info['video'] = video_result
        
        # Audio emotion (40% weight - highest because voice is very expressive)
        if audio_result and audio_result.get('emotion') and not audio_result.get('error'):
            confidence = audio_result.get('confidence', 0.5)
            # Adjust confidence based on audio quality
            duration = audio_result.get('audio_duration', 0)
            
            if duration > 2:  # Good duration for analysis
                confidence = min(confidence + 0.05, 1.0)
            
            emotions_data.append({
                'emotion': normalize_emotion(audio_result['emotion']),
                'confidence': confidence,
                'weight': 0.40,
                'modality': 'audio'
            })
            modality_info['audio'] = audio_result
        
        # Text emotion (25% weight)
        if text_result and text_result.get('emotion') and not text_result.get('error'):
            confidence = text_result.get('confidence', 0.5)
            text_length = len(text_result.get('original_text', ''))
            
            # Boost confidence for longer, more meaningful text
            if text_length > 20:
                confidence = min(confidence + 0.05, 1.0)
            
            emotions_data.append({
                'emotion': normalize_emotion(text_result['emotion']),
                'confidence': confidence,
                'weight': 0.25,
                'modality': 'text'
            })
            modality_info['text'] = text_result
        
        if not emotions_data:
            return {
                "emotion": "neutral", 
                "confidence": 0.5,
                "method": "default",
                "modalities_used": [],
                "individual_results": modality_info
            }
        
        # Method 1: Weighted confidence scoring
        emotion_scores = {}
        total_weight = 0
        
        for data in emotions_data:
            emotion = data['emotion']
            weighted_score = data['confidence'] * data['weight']
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0
            emotion_scores[emotion] += weighted_score
            total_weight += data['weight']
        
        # Method 2: Consensus-based approach
        emotion_votes = {}
        for data in emotions_data:
            emotion = data['emotion']
            confidence = data['confidence']
            
            # Weight votes by confidence
            vote_weight = confidence * data['weight']
            
            if emotion not in emotion_votes:
                emotion_votes[emotion] = 0
            emotion_votes[emotion] += vote_weight
        
        # Choose the emotion with highest weighted score
        if emotion_scores:
            best_emotion = max(emotion_scores.keys(), key=lambda x: emotion_scores[x])
            combined_confidence = emotion_scores[best_emotion] / total_weight if total_weight > 0 else 0.5
            
            # Boost confidence if multiple modalities agree
            agreeing_modalities = sum(1 for data in emotions_data if data['emotion'] == best_emotion)
            if agreeing_modalities > 1:
                combined_confidence = min(combined_confidence + 0.1, 1.0)
            
            # Ensure minimum confidence
            combined_confidence = max(combined_confidence, 0.4)
            
            return {
                "emotion": best_emotion,
                "confidence": combined_confidence,
                "method": "weighted_confidence",
                "modalities_used": [data['modality'] for data in emotions_data],
                "individual_results": modality_info,
                "emotion_scores": emotion_scores,
                "total_weight": total_weight
            }
        
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "method": "default",
            "modalities_used": [],
            "individual_results": modality_info
        }

# Initialize the emotion detector
detector = MultimodalEmotionDetector()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return app.send_static_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_emotions():
    """Main endpoint for multimodal emotion analysis"""
    try:
        start_time = time.time()
        
        # Get uploaded files and text
        video_file = request.files.get('video')
        audio_file = request.files.get('audio')
        text_content = request.form.get('text', '').strip()
        
        print(f"Received request with video: {video_file is not None}, audio: {audio_file is not None}, text: {bool(text_content)}")
        
        results = {
            'video': None,
            'audio': None, 
            'text': None,
            'combined': None,
            'processing_time': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Process video if provided
        if video_file and video_file.filename:
            try:
                # Save video file temporarily
                video_filename = secure_filename(f"video_{int(time.time())}.webm")
                video_path = os.path.join(detector.temp_dir, video_filename)
                video_file.save(video_path)
                
                print(f"Processing video: {video_path}")
                video_result = detector.detect_video_emotion(video_path)
                results['video'] = video_result
                
                # Clean up
                try:
                    os.remove(video_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"Error processing video: {e}")
                results['video'] = {"emotion": "neutral", "confidence": 0.5, "error": str(e)}
        
        # Process audio if provided
        if audio_file and audio_file.filename:
            try:
                # Save audio file temporarily
                audio_filename = secure_filename(f"audio_{int(time.time())}.webm")
                audio_path = os.path.join(detector.temp_dir, audio_filename)
                audio_file.save(audio_path)
                
                print(f"Processing audio: {audio_path}")
                audio_result = detector.detect_audio_emotion(audio_path)
                results['audio'] = audio_result
                
                # Clean up
                try:
                    os.remove(audio_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"Error processing audio: {e}")
                results['audio'] = {"emotion": "neutral", "confidence": 0.5, "error": str(e)}
        
        # Process text if provided
        if text_content:
            try:
                print(f"Processing text: {text_content[:50]}...")
                text_result = detector.detect_text_emotion(text_content)
                results['text'] = text_result
                
            except Exception as e:
                print(f"Error processing text: {e}")
                results['text'] = {"emotion": "neutral", "confidence": 0.5, "error": str(e)}
        
        # Combine results from all modalities
        try:
            combined_result = detector.combine_emotions(
                results['video'], 
                results['audio'], 
                results['text']
            )
            results['combined'] = combined_result
            
        except Exception as e:
            print(f"Error combining emotions: {e}")
            results['combined'] = {"emotion": "neutral", "confidence": 0.5, "error": str(e)}
        
        # Calculate processing time
        results['processing_time'] = round(time.time() - start_time, 2)
        
        print(f"Analysis completed in {results['processing_time']} seconds")
        print(f"Results: {results['combined']}")
        
        # Convert numpy types to JSON serializable types
        results = convert_numpy_types(results)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in analyze_emotions: {e}")
        error_results = {
            'error': str(e),
            'video': {"emotion": "neutral", "confidence": 0.5, "error": "Processing failed"},
            'audio': {"emotion": "neutral", "confidence": 0.5, "error": "Processing failed"},
            'text': {"emotion": "neutral", "confidence": 0.5, "error": "Processing failed"},
            'combined': {"emotion": "neutral", "confidence": 0.5, "error": "Processing failed"}
        }
        # Convert numpy types for error response too
        error_results = convert_numpy_types(error_results)
        return jsonify(error_results), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'text': detector.text_classifier is not None,
            'video': detector.face_cascade is not None,
            'audio': True  # Audio processing doesn't require pre-trained models
        }
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint for debugging"""
    return jsonify({
        'message': 'Multimodal Emotion Detection API is running!',
        'endpoints': {
            'analyze': '/analyze (POST) - Main emotion analysis endpoint',
            'health': '/health (GET) - Health check',
            'test': '/test (GET) - This endpoint'
        },
        'supported_formats': {
            'video': ['webm', 'mp4'],
            'audio': ['webm', 'wav', 'mp3'],
            'text': 'Plain text up to 1000 characters'
        }
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large. Please ensure your video/audio files are under the size limit.',
        'max_size': '16MB'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({
        'error': 'Internal server error occurred',
        'message': 'Please try again or contact support if the problem persists'
    }), 500

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def convert_numpy_types(obj):
        """Recursively convert NumPy data types in dicts/lists to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

if __name__ == '__main__':
    print("🚀 Starting Multimodal Emotion Detection Server...")
    print("📊 Loading models and initializing detector...")
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    print("✅ Server ready!")
    print("🌐 Access the application at: http://localhost:5000")
    print("📋 API endpoints:")
    print("   - POST /analyze - Main emotion analysis")
    print("   - GET /health - Health check")
    print("   - GET /test - Test endpoint")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )