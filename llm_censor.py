"""
LLM Censor module for SAM background removal quality control.

Uses Ollama vision models to:
1. Analyze original images and identify main objects
2. Evaluate segmentation quality
3. Suggest SAM parameter adjustments for better results
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
import io
import traceback

# Enable verbose logging
DEBUG = os.environ.get('STICKER_DEBUG', '0') == '1'

def log_debug(message):
    """Log debug message if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG LLM] {message}", file=sys.stderr)

def log_error(message, exc_info=None):
    """Log error message with optional exception info."""
    print(f"[ERROR LLM] {message}", file=sys.stderr)
    if exc_info:
        traceback.print_exception(*exc_info, file=sys.stderr)


class OllamaLLMCensor:
    """LLM censor for SAM segmentation quality control using Ollama."""
    
    def __init__(self, model_name='llava:13b', base_url='http://localhost:11434', enabled=True):
        """
        Initialize LLM censor.
        
        Args:
            model_name: Ollama model name (recommended: llava:13b, llava-next:latest)
            base_url: Ollama API URL (default: http://localhost:11434)
            enabled: Whether censor is enabled
        """
        self.enabled = enabled
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
        
        if enabled:
            self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check Ollama availability and model presence."""
        try:
            # Check Ollama availability
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama API unavailable: {response.status_code}")
            
            # Check if model exists
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            if self.model_name not in model_names:
                log_debug(f"Model {self.model_name} not found. Available models: {model_names}")
                print(f"[WARNING] Model {self.model_name} not found in Ollama.")
                print(f"Install it with: ollama pull {self.model_name}")
                print(f"Recommended models: llava:13b, llava-next:latest, llava:7b")
                self.enabled = False
            else:
                log_debug(f"Model {self.model_name} found and ready to use")
        except requests.exceptions.ConnectionError:
            log_error(f"Failed to connect to Ollama at {self.base_url}")
            print(f"[WARNING] Ollama unavailable. LLM censor disabled.")
            print(f"Make sure Ollama is running: ollama serve")
            self.enabled = False
        except Exception as e:
            log_error(f"Error checking Ollama: {e}")
            self.enabled = False
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            log_error(f"Error encoding image {image_path}: {e}")
            raise
    
    def _call_ollama_vision(self, image_path: str, prompt: str) -> str:
        """
        Call Ollama API with image.
        
        Args:
            image_path: Path to image
            prompt: Text prompt
            
        Returns:
            Model response
        """
        if not self.enabled:
            return ""
        
        try:
            # Encode image
            image_base64 = self._encode_image(image_path)
            
            # Prepare request for Ollama
            # Ollama uses special format for vision models
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }
            
            log_debug(f"Sending request to Ollama (model: {self.model_name})...")
            # Increased timeout for large images and slow models (5 minutes)
            # This prevents hanging on GPU-intensive vision models
            response = requests.post(self.api_url, json=payload, timeout=300)
            
            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                log_error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            answer = result.get('response', '').strip()
            log_debug(f"Received response from Ollama (length: {len(answer)} characters)")
            return answer
            
        except requests.exceptions.Timeout:
            log_error("Timeout when calling Ollama")
            raise Exception("Timeout when calling Ollama. Try increasing timeout or using a smaller model.")
        except Exception as e:
            log_error(f"Error calling Ollama: {e}")
            raise
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Try to extract JSON from text response."""
        # Try to find JSON in response
        text = text.strip()
        
        # Look for JSON block
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If JSON not found, try to extract information from text
        # This is a fallback for cases when model didn't return valid JSON
        result = {}
        
        # Look for keywords
        text_lower = text.lower()
        if 'approved' in text_lower or 'good' in text_lower or 'excellent' in text_lower:
            result['approved'] = True
        elif 'rejected' in text_lower or 'bad' in text_lower or 'poor' in text_lower:
            result['approved'] = False
        
        # Check for wrong selection (background kept instead of object)
        if 'wrong' in text_lower or ('background' in text_lower and 'kept' in text_lower):
            result['wrong_selection'] = True
            result['approved'] = False
        
        # Look for numeric values for parameters
        import re
        if 'points_per_side' in text_lower:
            match = re.search(r'points_per_side[:\s]+(\d+)', text_lower)
            if match:
                result['suggested_points_per_side'] = int(match.group(1))
        
        if 'iou' in text_lower or 'pred_iou_thresh' in text_lower:
            match = re.search(r'(?:iou|pred_iou_thresh)[:\s]+([\d.]+)', text_lower)
            if match:
                result['suggested_iou_thresh'] = float(match.group(1))
        
        if 'stability' in text_lower or 'stability_score' in text_lower:
            match = re.search(r'stability[:\s]+([\d.]+)', text_lower)
            if match:
                result['suggested_stability'] = float(match.group(1))
        
        return result if result else {'approved': True, 'reason': 'Failed to parse response'}
    
    def analyze_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze original image and identify main object.
        
        Args:
            image_path: Path to original image
            
        Returns:
            Dictionary with image analysis or None if censor is disabled
        """
        if not self.enabled:
            return None
        
        log_debug(f"Analyzing original image: {image_path}")
        
        prompt = """Analyze this image and provide a JSON response with:
{
  "main_subject": "description of the main object/subject",
  "expected_boundaries": "description of expected object boundaries",
  "background_characteristics": "description of background",
  "challenging_areas": "any areas that might be difficult to segment (transparency, similar colors, etc.)",
  "image_type": "photo/illustration/logo/etc"
}

Provide ONLY valid JSON, no additional text."""
        
        try:
            response = self._call_ollama_vision(image_path, prompt)
            analysis = self._parse_json_response(response)
            log_debug(f"Analysis complete: {analysis}")
            return analysis
        except Exception as e:
            log_error(f"Error analyzing image: {e}")
            return None
    
    def select_best_mask(self, original_path: str, mask_results: List[Dict[str, Any]]) -> Optional[int]:
        """
        Select the best mask from multiple candidates using LLM.
        
        Args:
            original_path: Path to original image
            mask_results: List of dicts with 'mask_index', 'result_path', 'area', 'predicted_iou'
            
        Returns:
            Index of best mask or None if selection failed
        """
        if not self.enabled or not mask_results:
            return None
        
        log_debug(f"LLM selecting best mask from {len(mask_results)} candidates...")
        
        try:
            # Create comparison image with all mask results
            original_img = Image.open(original_path).convert('RGB')
            width = original_img.width
            height = original_img.height
            
            # Create grid of results
            cols = min(3, len(mask_results))
            rows = (len(mask_results) + cols - 1) // cols
            
            grid_width = width * cols
            grid_height = height * (rows + 1)  # +1 for original
            
            combined = Image.new('RGB', (grid_width, grid_height), color='white')
            
            # Place original at top
            combined.paste(original_img, (0, 0))
            
            # Place mask results in grid
            for idx, mask_result in enumerate(mask_results):
                result_img = Image.open(mask_result['result_path']).convert('RGBA')
                result_rgb = result_img.convert('RGB')
                
                col = idx % cols
                row = idx // cols
                x = col * width
                y = height + row * height
                combined.paste(result_rgb, (x, y))
            
            # Save temporary comparison image
            temp_path = Path(original_path).parent / f"_temp_mask_selection_{Path(original_path).stem}.png"
            combined.save(temp_path)
            
            prompt = f"""The top image is the original. Below are {len(mask_results)} segmentation results (numbered left to right, top to bottom, starting from 0).

CRITICAL: Your task is to identify which result BEST removes the BACKGROUND while KEEPING the MAIN OBJECT/SUBJECT.

IMPORTANT RULES:
- If a result keeps the background and removes the object, it is WRONG
- The correct result should show the main object/subject with transparent background
- Choose the result where the main character/object is fully visible and background is removed

Provide a JSON response:
{{
  "best_mask_index": 0-{len(mask_results)-1} (0-based index, left to right, top to bottom),
  "reason": "why this mask is best",
  "issues_with_others": ["what's wrong with other masks"]
}}

Provide ONLY valid JSON, no additional text."""
            
            response = self._call_ollama_vision(str(temp_path), prompt)
            selection = self._parse_json_response(response)
            
            # Delete temporary file
            try:
                temp_path.unlink()
            except:
                pass
            
            best_idx = selection.get('best_mask_index')
            if best_idx is not None and 0 <= best_idx < len(mask_results):
                log_debug(f"LLM selected mask index {best_idx}")
                return best_idx
            else:
                log_debug(f"LLM selection invalid: {best_idx}, using first mask")
                return 0
                
        except Exception as e:
            log_error(f"Error selecting best mask: {e}")
            return 0  # Fallback to first mask
    
    def evaluate_segmentation(self, original_path: str, result_path: str) -> Dict[str, Any]:
        """
        Evaluate segmentation quality by comparing original and result.
        
        Args:
            original_path: Path to original image
            result_path: Path to segmentation result
            
        Returns:
            Dictionary with quality evaluation
        """
        if not self.enabled:
            return {'approved': True, 'reason': 'LLM censor disabled'}
        
        log_debug(f"Evaluating segmentation quality: {original_path} -> {result_path}")
        
        # Create combined image for comparison
        try:
            original_img = Image.open(original_path).convert('RGB')
            result_img = Image.open(result_path).convert('RGBA')
            
            # Create side-by-side image
            width = max(original_img.width, result_img.width)
            height = original_img.height + result_img.height + 20
            
            combined = Image.new('RGB', (width, height), color='white')
            combined.paste(original_img, (0, 0))
            combined.paste(result_img.convert('RGB'), (0, original_img.height + 20))
            
            # Save temporary image
            temp_path = Path(result_path).parent / f"_temp_comparison_{Path(result_path).stem}.png"
            combined.save(temp_path)
            
            prompt = """Compare the original image (top) with the segmentation result (bottom).

IMPORTANT: The goal is to REMOVE THE BACKGROUND and KEEP THE MAIN OBJECT/SUBJECT. 
If the background is kept and the object is removed, this is WRONG.

Evaluate the segmentation quality and provide a JSON response:
{
  "approved": true/false,
  "quality_score": 1-10,
  "issues": ["list of specific problems found"],
  "main_object_captured": true/false,
  "background_artifacts": true/false,
  "missing_parts": true/false,
  "wrong_selection": true/false (true if background was kept instead of object),
  "suggestions": {
    "points_per_side": number (if needs more detail, increase; if too many masks, decrease),
    "pred_iou_thresh": 0.0-1.0 (if masks are too loose, increase; if too strict, decrease),
    "stability_score_thresh": 0.0-1.0 (if masks are unstable, increase),
    "min_mask_region_area": number (if small artifacts, increase; if object is filtered out, decrease),
    "crop_n_layers": 0-3 (if object is missed, increase; if too many false positives, decrease),
    "crop_n_points_downscale_factor": 1-4 (if object is missed, decrease; if too slow, increase),
    "box_nms_thresh": 0.1-0.9 (if too many overlapping masks, increase; if object is filtered, decrease)
  },
  "reason": "explanation of the evaluation"
}

Provide ONLY valid JSON, no additional text."""
            
            response = self._call_ollama_vision(str(temp_path), prompt)
            evaluation = self._parse_json_response(response)
            
            # Delete temporary file
            try:
                temp_path.unlink()
            except:
                pass
            
            log_debug(f"Evaluation complete: {evaluation}")
            return evaluation
            
        except Exception as e:
            log_error(f"Error evaluating segmentation: {e}")
            return {'approved': True, 'reason': f'Evaluation error: {e}'}
    
    def suggest_sam_parameters(self, evaluation: Dict[str, Any], current_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Suggest new SAM parameters based on evaluation.
        
        Args:
            evaluation: Segmentation evaluation result
            current_params: Current SAM parameters
            
        Returns:
            Dictionary with new parameters or None if no changes needed
        """
        if not self.enabled or evaluation.get('approved', False):
            return None
        
        suggestions = evaluation.get('suggestions', {})
        if not suggestions:
            return None
        
        # Apply suggestions with reasonable limits
        new_params = current_params.copy()
        
        if 'points_per_side' in suggestions:
            # Limit range 16-64
            new_params['points_per_side'] = max(16, min(64, int(suggestions['points_per_side'])))
        
        if 'pred_iou_thresh' in suggestions:
            # Limit range 0.5-0.99
            new_params['pred_iou_thresh'] = max(0.5, min(0.99, float(suggestions['pred_iou_thresh'])))
        
        if 'stability_score_thresh' in suggestions:
            # Limit range 0.5-0.99
            new_params['stability_score_thresh'] = max(0.5, min(0.99, float(suggestions['stability_score_thresh'])))
        
        if 'min_mask_region_area' in suggestions:
            # Limit range 0-1000
            new_params['min_mask_region_area'] = max(0, min(1000, int(suggestions['min_mask_region_area'])))
        
        if 'crop_n_layers' in suggestions:
            # Limit range 0-3
            new_params['crop_n_layers'] = max(0, min(3, int(suggestions['crop_n_layers'])))
        
        if 'crop_n_points_downscale_factor' in suggestions:
            # Limit range 1-4
            new_params['crop_n_points_downscale_factor'] = max(1, min(4, int(suggestions['crop_n_points_downscale_factor'])))
        
        if 'box_nms_thresh' in suggestions:
            # Limit range 0.1-0.9
            new_params['box_nms_thresh'] = max(0.1, min(0.9, float(suggestions['box_nms_thresh'])))
        
        log_debug(f"New parameters suggested: {new_params}")
        return new_params


def create_llm_censor(model_name='llava:13b', base_url='http://localhost:11434', enabled=True) -> Optional[OllamaLLMCensor]:
    """
    Create LLM censor instance.
    
    Args:
        model_name: Ollama model name
        base_url: Ollama API URL
        enabled: Whether censor is enabled
        
    Returns:
        OllamaLLMCensor instance or None if creation failed
    """
    if not enabled:
        return None
    
    try:
        return OllamaLLMCensor(model_name=model_name, base_url=base_url, enabled=enabled)
    except Exception as e:
        log_error(f"Failed to create LLM censor: {e}")
        return None

