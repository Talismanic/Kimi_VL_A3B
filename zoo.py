import os
import re
import logging
import json
from PIL import Image
from typing import Dict, Any, List, Union, Optional

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

DEFAULT_DETECTION_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in detecting and localizating meaningful visual elements. 

You can detect and localize objects, components, people, places, things, and UI elements in images using 2D bound boxes.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "detections": [
        {
            "bbox_2d": [x1, y1, x2, y2],
            "label": "descriptive label for the bounding box"
        },
        {
            "bbox_2d": [x1, y1, x2, y2],
            "label": "descriptive label for the bounding box"
        }
    ]
}
```

The JSON should contain bounding boxes in pixel coordinates [x1,y1,x2,y2] format, where:
- x1,y1 is the top-left corner
- x2,y2 is the bottom-right corner
- Provide specific, descriptive labels for each detected element
- Include all relevant elements that match the user's request
- For UI elements, include their function when possible (e.g., "Login Button" rather than just "Button")
- If many similar elements exist, prioritize the most prominent or relevant ones

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and detect.

Think step by step.
"""

DEFAULT_KEYPOINT_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in key point detection across any visual domain. A key point represents the center of any meaningful visual element. 

Key points should adapt to the context (physical world, digital interfaces, UI elements, etc.) while maintaining consistent accuracy and relevance. 

For each key point identify the key point and provide a contextually appropriate label and always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "keypoints": [
        {
            "point_2d": [x, y],
            "label": "descriptive label for the point"
        }
    ]
}
```

The JSON should contain points in pixel coordinates [x,y] format, where:
- x is the horizontal center coordinate of the visual element
- y is the vertical center coordinate of the visual element
- Include all relevant elements that match the user's request
- You can point to multiple visual elements

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and point.

Think step by step
"""

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You are a helpful assistant. You specializes in comprehensive classification across any visual domain, capable of analyzing:

Unless specifically requested for single-class output, multiple relevant classifications can be provided.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "classifications": [
        {
            "label": "descriptive class label"
        }
    ]
}
```

The JSON should contain a list of classifications where:
- Each classification must have a 'label' field
- Labels should be descriptive strings describing what you've identified in the image
- The response should be a list of classifications

Think step by step
"""
   
DEFAULT_OCR_SYSTEM_PROMPT = """You are a helpful assistant specializing in text detection and recognition (OCR) in images. Your can read, detect, and locate text from any visual content, including documents, UI elements, signs, or any other text-containing regions.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "text_detections": [
        {
            "bbox_2d": [x1, y1, x2, y2],  // Coordinates: [top-left x, top-left y, bottom-right x, bottom-right y]
            "text_type": "title|abstract|heading|paragraph|button|link|label|title|menu_item|input_field|icon|list_item|etc.",  // Select appropriate text category
            "text": "Exact text content found in this region"  // Transcribe text exactly as it appears
        }
    ]
}
```

The JSON should contain bounding boxes in pixel coordinates [x1,y1,x2,y2] format, where:
- x1,y1 is the top-left corner
- x2,y2 is the bottom-right corner
- 'text_type' is important to get right, it's the text region category based on the document, including but not limited to: title, abstract, heading, paragraph, button, link, label, icon, menu item, etc.
- The 'text' field should be a string containing the exact text content found in the region

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's perform the OCR detections.

Think step by step
"""

DEFAULT_VQA_SYSTEM_PROMPT = "You are a helpful assistant. You provide clear and concise answerss to questions about images. Report answers in natural language text in English."

DEFAULT_AGENTIC_PROMPT = """You are a helpful assistant that generates PyAutoGUI code to interact with elements in images. When shown a screenshot, provide the exact Python code needed to interact with requested elements.

Example response:
```python
pyautogui.click(x=100, y=200)
```

Example PyAutoGUI commands include:
Mouse:
- click(x, y)
- doubleClick(x, y)
- rightClick(x, y)
- moveTo(x, y)
- dragTo(x, y)
- mouseDown(x, y)
- mouseUp(x, y)

Keyboard:
- typewrite('text')
- press('key')
- hotkey('ctrl', 'c')
- keyDown('key')
- keyUp('key')

Scrolling:
- scroll(amount)  # Positive to scroll up, negative to scroll down
- hscroll(amount) # Horizontal scroll

Screen:
- screenshot()
- pixel(x, y)     # Get color of pixel
- locateOnScreen('image.png')
- center(box)     # Get center of a region

Please observe the screenshot, please locate the following elements with action and point.<instruction> 
"""

OPERATIONS = {
    "detect": DEFAULT_DETECTION_SYSTEM_PROMPT,
    "point": DEFAULT_KEYPOINT_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
    "ocr": DEFAULT_OCR_SYSTEM_PROMPT,
    "agentic": DEFAULT_AGENTIC_PROMPT,
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class KimiVLModel(SamplesMixin, Model):
    """A FiftyOne model for running Qwen-VL vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        model_kwargs = {
            "device_map":self.device,
            }

        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Only set specific torch_dtype for CUDA devices
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
        )

        logger.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True
        )

        self.model.eval()
    
    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    
    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else OPERATIONS[self.operation]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    def _parse_json(self, s: str) -> Optional[Dict]:
        """Parse JSON from model output and extract reasoning information.
        
        This method handles multiple formats of model output:
        1. Text with ◁think▷ tags containing reasoning
        2. JSON within markdown code blocks (```json)
        3. Raw JSON strings
        
        Args:
            s: Raw string output from the model containing JSON and possibly reasoning
                
        Returns:
            Dict: Dictionary containing:
                - data: The parsed JSON content
                - reasoning: Text extracted from ◁think▷ tags
            None: If parsing fails
            Original input: If input is not a string
        
        Example input:
            "◁think▷This image contains a person◁/think▷
            ```json
            {"detections": [{"bbox": [0,0,100,100], "label": "person"}]}
            ```"
        """
        # Return non-string inputs as-is
        if not isinstance(s, str):
            return s
        
        # Extract reasoning from between ◁think▷ tags if present
        reasoning = ""
        if "◁think▷" in s and "◁/think▷" in s:
            try:
                # Split on tags and take content between them
                reasoning = s.split("◁think▷")[1].split("◁/think▷")[0].strip()
            except:
                logger.debug("Failed to extract reasoning from ◁think▷ tags")
        
        # Extract JSON content from markdown code blocks if present
        if "```json" in s:
            try:
                # Split on markdown markers and take JSON content
                json_str = s.split("```json")[1].split("```")[0].strip()
            except:
                json_str = s
        else:
            json_str = s
            
        # Attempt to parse the JSON string
        try:
            parsed_json = json.loads(json_str)
            return {
                "data": parsed_json,  # The actual JSON content
                "reasoning": reasoning  # The extracted reasoning text
            }
        except:
            # Log parsing failures for debugging
            logger.debug(f"Failed to parse JSON: {json_str[:200]}")
            return None

    def _to_detections(
        self, 
        boxes: List[Dict], 
        image_width: int, 
        image_height: int, 
        image_grid_thw: torch.Tensor = None, 
        patch_size: int = 14
    ) -> fo.Detections:
        """Convert bounding boxes to FiftyOne Detections with associated reasoning.
        
        Takes detection results and converts them to FiftyOne's format, including:
        - Coordinate normalization
        - Label extraction  
        - Reasoning attachment
        
        Args:
            boxes: Detection results, either:
                - List of detection dictionaries
                - Dictionary containing 'data' and 'reasoning'
            image_width: Width of the original image in pixels
            image_height: Height of the original image in pixels
            image_grid_thw: Optional tensor with processed image grid dimensions
            patch_size: Size of each patch (default 14)
        
        Returns:
            fo.Detections object with normalized coordinates and reasoning
        """
        detections = []
        
        # Extract reasoning if present in dictionary format
        reasoning = boxes.get("reasoning", "") if isinstance(boxes, dict) else ""
        
        # Handle nested dictionary structures
        if isinstance(boxes, dict):
            # Try to get data field, fall back to original dict if not found
            boxes = boxes.get("data", boxes)
            if isinstance(boxes, dict):
                # If still a dict, try to find first list value
                boxes = next((v for v in boxes.values() if isinstance(v, list)), boxes)
        
        # Ensure boxes is a list, even for single box input
        boxes = boxes if isinstance(boxes, list) else [boxes]
        
        for box in boxes:
            try:
                # Try to get bbox from either bbox_2d or bbox field
                bbox = box.get('bbox_2d', box.get('bbox', None))
                if not bbox:
                    continue
                
                # If image_grid_thw is provided, convert normalized coords to pixel coords
                if image_grid_thw is not None:
                    # Extract grid dimensions (THW order)
                    proc_height = float(image_grid_thw[0][0].cpu() * patch_size)
                    proc_width = float(image_grid_thw[0][1].cpu() * patch_size)
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1 = bbox[0] * proc_width
                    y1 = bbox[1] * proc_height
                    x2 = bbox[2] * proc_width
                    y2 = bbox[3] * proc_height
                    
                    # Update bbox to pixel coordinates for conversion
                    bbox = [x1, y1, x2, y2]
                
                # Convert pixel coordinates to normalized [0,1] coordinates
                x1, y1, x2, y2 = map(float, bbox)
                x = x1 / image_width  # Normalized left x
                y = y1 / image_height # Normalized top y
                w = (x2 - x1) / image_width  # Normalized width
                h = (y2 - y1) / image_height # Normalized height
                
                # Create Detection object with normalized coordinates
                detection = fo.Detection(
                    label=str(box.get("label", "object")),
                    bounding_box=[x, y, w, h],
                    reasoning=reasoning  # Attach reasoning to detection
                )
                detections.append(detection)
                
            except Exception as e:
                # Optionally log errors
                logger.debug(f"Error processing box {box}: {e}")
                continue
                
        return fo.Detections(detections=detections)

    def _to_ocr_detections(
        self, 
        boxes: List[Dict], 
        image_width: int, 
        image_height: int, 
        image_grid_thw: torch.Tensor = None, 
        patch_size: int = 14
    ) -> fo.Detections:
        """Convert OCR results to FiftyOne Detections with reasoning.
        
        Takes OCR detection results and converts them to FiftyOne's format, including:
        - Coordinate normalization
        - Text content preservation
        - Text type categorization
        - Reasoning attachment
        
        Args:
            boxes: OCR detection results, either:
                - List of OCR dictionaries
                - Dictionary containing 'data' and 'reasoning'
            image_width: Width of the original image in pixels
            image_height: Height of the original image in pixels
            image_grid_thw: Optional tensor with processed image grid dimensions
            patch_size: Size of each patch (default 14)
            
        Returns:
            fo.Detections object containing the OCR annotations with text content and reasoning
        """
        detections = []
        
        # Extract reasoning if present in dictionary format
        reasoning = boxes.get("reasoning", "") if isinstance(boxes, dict) else ""
        
        # Handle nested dictionary structures
        if isinstance(boxes, dict):
            # Try to get data field, fall back to original dict if not found
            boxes = boxes.get("data", boxes)
            if isinstance(boxes, dict):
                # If still a dict, try to find first list value (usually "text_detections")
                boxes = next((v for v in boxes.values() if isinstance(v, list)), boxes)
        
        # Ensure boxes is a list, even for single box input
        boxes = boxes if isinstance(boxes, list) else [boxes]
        
        # Process each OCR box
        for box in boxes:
            try:
                # Extract the bounding box coordinates and text content
                bbox = box.get('bbox_2d', box.get('bbox', None))
                text = box.get('text')  # The actual text string
                text_type = box.get('text_type', 'text')  # Type of text, defaults to 'text'
                
                # Skip if missing required bbox or text fields
                if not bbox or not text:
                    continue
                
                # If image_grid_thw is provided, convert normalized coords to pixel coords
                if image_grid_thw is not None:
                    # Extract grid dimensions (THW order)
                    proc_height = float(image_grid_thw[0][0].cpu() * patch_size)
                    proc_width = float(image_grid_thw[0][1].cpu() * patch_size)
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1 = bbox[0] * proc_width
                    y1 = bbox[1] * proc_height
                    x2 = bbox[2] * proc_width
                    y2 = bbox[3] * proc_height
                    
                    # Update bbox to pixel coordinates for conversion
                    bbox = [x1, y1, x2, y2]
                
                # Convert pixel coordinates to normalized [0,1] coordinates
                x1, y1, x2, y2 = map(float, bbox)
                x = x1 / image_width  # Normalized left x
                y = y1 / image_height # Normalized top y
                w = (x2 - x1) / image_width  # Normalized width
                h = (y2 - y1) / image_height # Normalized height
                
                # Create Detection object with normalized coordinates
                detection = fo.Detection(
                    label=str(text_type),
                    bounding_box=[x, y, w, h],
                    text=str(text),
                    reasoning=reasoning  # Attach reasoning to detection
                )
                detections.append(detection)
                
            except Exception as e:
                # Log any errors processing individual boxes but continue
                logger.debug(f"Error processing OCR box {box}: {e}")
                continue
                
        # Return all detections wrapped in a FiftyOne Detections container
        return fo.Detections(detections=detections)
    
    def _to_keypoints(self, points: List[Dict], image_width: int, image_height: int, image_grid_thw: torch.Tensor = None, patch_size: int = 14) -> fo.Keypoints:
        """Convert keypoint detections to FiftyOne Keypoints with reasoning.
        
        Processes keypoint coordinates and normalizes them to [0,1] range while
        preserving associated labels and reasoning.
        
        Args:
            points: Keypoint detection results, either:
                - List of keypoint dictionaries
                - Dictionary containing 'data' and 'reasoning'
            image_width: Original image width in pixels
            image_height: Original image height in pixels
            image_grid_thw: Optional tensor with processed image grid dimensions
            patch_size: Size of each patch (default 14)
            
        Returns:
            fo.Keypoints object containing the keypoint annotations with reasoning
            
        Example input:
            {
                "data": [{"point_2d": [100,100], "label": "nose"}],
                "reasoning": "Identified facial features"
            }
        """
        keypoints = []
        
        # Extract reasoning if present
        reasoning = points.get("reasoning", "") if isinstance(points, dict) else ""
        
        # Handle nested dictionary structures
        if isinstance(points, dict):
            points = points.get("data", points)
            if isinstance(points, dict):
                points = next((v for v in points.values() if isinstance(v, list)), points)
        
        # Process each keypoint
        for point in points:
            try:
                # Extract the x,y coordinates from the point_2d field
                point_2d = point["point_2d"]
                
                # If coordinates are PyTorch tensors, convert them to Python floats
                if torch.is_tensor(point_2d[0]):
                    point_2d = [float(p.cpu()) for p in point_2d]  # Move to CPU and convert to float
                
                # If image_grid_thw is provided, convert normalized coords to pixel coords
                if image_grid_thw is not None:
                    # Extract grid dimensions (THW order)
                    proc_height = float(image_grid_thw[0][0].cpu() * patch_size)
                    proc_width = float(image_grid_thw[0][1].cpu() * patch_size)
                    
                    # Convert normalized coordinates to pixel coordinates
                    x = point_2d[0] * proc_width
                    y = point_2d[1] * proc_height
                    
                    # Update point_2d to pixel coordinates for normalization
                    point_2d = [x, y]
                
                # Normalize coordinates to [0,1] range
                normalized_point = [
                    point_2d[0] / image_width,
                    point_2d[1] / image_height
                ]
                
                # Create a FiftyOne Keypoint object with label and coordinates
                keypoint = fo.Keypoint(
                    label=str(point.get("label", "point")),  # Convert label to string for consistency
                    points=[normalized_point],  # Wrap point_2d in list since Keypoint expects list of points
                    reasoning=reasoning  # Attach reasoning to keypoint
                )
                keypoints.append(keypoint)  # Add the keypoint to our collection
                
            except Exception as e:
                # Log any errors but continue processing remaining points
                logger.debug(f"Error processing point {point}: {e}")
                continue
                
        # Return all keypoints wrapped in a FiftyOne Keypoints container
        return fo.Keypoints(keypoints=keypoints)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert classification results to FiftyOne Classifications with reasoning.
        
        Processes classification labels and associated reasoning into FiftyOne's format.
        
        Args:
            classes: Classification results, either:
                - List of classification dictionaries
                - Dictionary containing 'data' and 'reasoning'
                
        Returns:
            fo.Classifications object containing the converted classification 
            annotations with labels, optional confidence scores, and reasoning
            
        Example input:
            {
                "data": [{"label": "cat"}, {"label": "animal"}],
                "reasoning": "Image shows a domestic cat"
            }
        """
        classifications = []
        
        # Extract reasoning if present
        reasoning = classes.get("reasoning", "") if isinstance(classes, dict) else ""
        
        # Handle nested dictionary structures
        if isinstance(classes, dict):
            classes = classes.get("data", classes)
            if isinstance(classes, dict):
                classes = next((v for v in classes.values() if isinstance(v, list)), classes)
        
        # Process each classification dictionary
        for cls in classes:
            try:
                # Create Classification object with required label and optional confidence
                classification = fo.Classification(
                    label=str(cls["label"]),  # Convert label to string for consistency
                    reasoning=reasoning  # Attach reasoning to classification
                )
                classifications.append(classification)
            except Exception as e:
                # Log any errors but continue processing remaining classifications
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        # Return Classifications container with all processed results
        return fo.Classifications(classifications=classifications)
    
    def _to_agentic_keypoints(self, output_text: str, image_width: int, image_height: int, image_grid_thw: torch.Tensor = None, patch_size: int = 14) -> fo.Keypoints:
        """Convert agentic PyAutoGUI code to FiftyOne Keypoints.
        
        Parses PyAutoGUI code snippets to extract coordinates and creates keypoints
        with the full code as the label.
        
        Args:
            output_text: Raw output text containing PyAutoGUI code and possibly reasoning
            image_width: Original image width in pixels
            image_height: Original image height in pixels
            image_grid_thw: Optional tensor with processed image grid dimensions
            patch_size: Size of each patch (default 14)
            
        Returns:
            fo.Keypoints object containing the agentic action keypoints
            
        Example input:
            "◁think▷I need to click the login button◁/think▷
            ```python
            pyautogui.click(x=0.972, y=0.186)
            ```"
        """
        import re
        
        keypoints = []
        
        # Extract reasoning if present
        reasoning = ""
        if "◁think▷" in output_text and "◁/think▷" in output_text:
            try:
                reasoning = output_text.split("◁think▷")[1].split("◁/think▷")[0].strip()
            except:
                logger.debug("Failed to extract reasoning from ◁think▷ tags")
        
        # Extract Python code blocks
        python_blocks = []
        if "```python" in output_text:
            # Find all python code blocks
            pattern = r'```python\n(.*?)\n```'
            matches = re.findall(pattern, output_text, re.DOTALL)
            python_blocks.extend(matches)
        
        # Process each code block to find coordinates
        for code_block in python_blocks:
            try:
                # Look for pyautogui commands with x, y coordinates
                # Patterns to match: pyautogui.click(x=0.972, y=0.186) or pyautogui.click(0.972, 0.186)
                coord_patterns = [
                    r'pyautogui\.\w+\([^)]*x\s*=\s*([\d.]+)[^)]*y\s*=\s*([\d.]+)[^)]*\)',
                    r'pyautogui\.\w+\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)',
                ]
                
                for pattern in coord_patterns:
                    matches = re.findall(pattern, code_block)
                    for match in matches:
                        x_coord, y_coord = map(float, match)
                        
                        # If image_grid_thw is provided, convert normalized coords to pixel coords
                        if image_grid_thw is not None:
                            # Extract grid dimensions (THW order)
                            proc_height = float(image_grid_thw[0][0].cpu() * patch_size)
                            proc_width = float(image_grid_thw[0][1].cpu() * patch_size)
                            
                            # Convert normalized coordinates to pixel coordinates
                            x_coord = x_coord * proc_width
                            y_coord = y_coord * proc_height
                        
                        # Check if coordinates need normalization (if they're > 1, they're likely pixel coords)
                        if x_coord > 1.0 or y_coord > 1.0:
                            # Convert from pixel coordinates to normalized
                            normalized_point = [x_coord / image_width, y_coord / image_height]
                        else:
                            # Already normalized or very small coordinates
                            normalized_point = [x_coord, y_coord]
                        
                        # Create keypoint with the full code block as label
                        keypoint = fo.Keypoint(
                            label=code_block.strip(),
                            points=[normalized_point],
                            reasoning=reasoning
                        )
                        keypoints.append(keypoint)
                        
            except Exception as e:
                logger.debug(f"Error processing code block {code_block}: {e}")
                continue
        
        return fo.Keypoints(keypoints=keypoints)


    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Keypoints, fo.Classifications, str]:
        """Process a single image through the model and return predictions.
        
        This internal method handles the core prediction logic including:
        - Constructing the chat messages with system prompt and user query
        - Processing the image and text through the model
        - Parsing the output based on the operation type (detection/points/classification/VQA)
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            One of:
            - fo.Detections: For object detection results
            - fo.Keypoints: For keypoint detection results  
            - fo.Classifications: For classification results
            - str: For VQA text responses
            
        Raises:
            ValueError: If no prompt has been set
        """
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt  # Start with instance default
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)  # Local variable, doesn't affect instance
        
        if not prompt:
            raise ValueError("No prompt provided.")
        
        messages = [
            {
                "role": "system", 
                "content": [  
                    {
                        "type": "text",
                        "text": self.system_prompt
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image", 
                        "image": sample.filepath if sample else image
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True
            )
        
        inputs = self.processor(
            text=[text], 
            images=[image], 
            padding=True, 
            truncation=True,
            return_tensors="pt").to(self.device)
        
        # Set recommended temperature based on model type per the model card
        temperature = 0.2 if self.model_path == "moonshotai/Kimi-VL-A3B-Instruct" else 0.6
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=8192,
                do_sample=True,
                temperature=temperature,
                )

        generated_ids_trimmed = [generated_ids[len(input_ids):] for input_ids, generated_ids in zip(inputs.input_ids, generated_ids)]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)[0]

        # Get image dimensions from PIL image
        image_width, image_height = image.size

        # For VQA, return the raw text output
        if self.operation == "vqa":
            return output_text.strip()
        # For agentic, parse PyAutoGUI code and convert to keypoints
        elif self.operation == "agentic":
            return self._to_agentic_keypoints(output_text, image_width, image_height, inputs['image_grid_hws'])
        elif self.operation == "detect":
            parsed_output = self._parse_json(output_text)
            return self._to_detections(
                parsed_output, 
                image_width=image_width, 
                image_height=image_height,
                image_grid_thw=inputs['image_grid_hws']
                )
        elif self.operation == "point":
            parsed_output = self._parse_json(output_text)
            return self._to_keypoints(parsed_output, image_width, image_height, inputs['image_grid_hws'])
        elif self.operation == "classify":
            parsed_output = self._parse_json(output_text)
            return self._to_classifications(parsed_output)
        elif self.operation == "ocr":
            parsed_output = self._parse_json(output_text)
            return self._to_ocr_detections(
                parsed_output, 
                image_width=image_width, 
                image_height=image_height,
                image_grid_thw=inputs['image_grid_hws']
                )

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)