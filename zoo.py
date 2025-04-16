
import os
import logging
import json
from PIL import Image
from typing import Dict, Any, List, Union, Optional

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)

DEFAULT_DETECTION_SYSTEM_PROMPT = """You are a helpful assistant. The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions, which will request that you detect both primary elements and their associated components. 

You specialize in detection and localization of any meaningful visual elements. You can identify and localize objects, components, people, places, things, and UI elements in images.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "label": "descriptive label for the action that can be taken or object detected"
        },
        {
            "bbox_2d": [x1, y1, x2, y2],
            "label": "descriptive label for the action that can be taken or object detected"
        }
    ]
}
```

The JSON should contain bounding boxes in pixel coordinates [x1,y1,x2,y2] format, where:
- x1,y1 is the top-left corner
- x2,y2 is the bottom-right corner
- Use descriptive, context-specific labels. 
- Each detection should have a 'bbox' (or 'bbox_2d') and optional 'label' field
- The 'label' field should be a string describing the detected object or action
- Once something has been localized, do not localize it again and move on to the the next object

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and do what they tell you to.
"""

DEFAULT_KEYPOINT_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in key point detection across any visual domain. A key point represents the center of any meaningful visual element. 

Key points should adapt to the context (physical world, digital interfaces, UI elements, etc.) while maintaining consistent accuracy and relevance. 

For each key point:
1. Identify the key point and provide a contextually appropriate label
2. Locate the center of the key point 

Always return your response as valid JSON wrapped in ```json blocks.

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
- x is the horizontal coordinate from left edge
- y is the vertical coordinate from top edge
- Each point must have a 'point_2d' field with [x,y] coordinates
- Each point should have a descriptive 'label' field of what the point represents
- Once something has been pointed, do not point at it again and move on to the the next object

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and do what they tell you to.
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
- Once you have determined a classification, do not repeat it and move on to the next classification

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and do what they tell you to.
"""
   
DEFAULT_OCR_SYSTEM_PROMPT = """You are a helpful assistant specializing in text detection and recognition (OCR) in images. Your task is to locate and read text from any visual content, including documents, UI elements, signs, or any other text-containing regions.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "text_detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "text": "exact text content found in this region",
            "text_type": "the text region category based on the document, including but not limited to: title, abstract, heading, paragraph, button, link, label, icon, menu item, etc.",
        }
    ]
}
```

The JSON should contain bounding boxes in pixel coordinates [x1,y1,x2,y2] format, where:
- x1,y1 is the top-left corner
- x2,y2 is the bottom-right corner
- All coordinates are in absolute pixel values
- text_type is the text region category based on the document, including but not limited to: title, abstract, heading, paragraph, button, link, label, icon, menu item, etc.
- The 'text' field should be a string containing the exact text content found in the region preserving:
  - Case sensitivity
  - Numbers and special characters
  - Line breaks (if present)
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

Please observe the screenshot, and return the command which fulfils the user's request.
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

        # Set dtype for CUDA devices
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else None
        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        if self.torch_dtype:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                # local_files_only=True,
                device_map=self.device,
                torch_dtype=self.torch_dtype
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                # local_files_only=True,
                device_map=self.device,
            )
        
        logger.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True
        )

        self.model.eval()

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
        """Parse JSON from model output.
        
        The model may return JSON in different formats:
        1. Raw JSON string
        2. JSON wrapped in markdown code blocks (```json ... ```)
        3. Non-JSON string (returns None)
        
        Args:
            s: String output from the model to parse
            
        Returns:
            Dict: Parsed JSON dictionary if successful
            None: If parsing fails or input is invalid
            Original input: If input is not a string
        """
        # Return input directly if not a string
        if not isinstance(s, str):
            return s
            
        # Handle JSON wrapped in markdown code blocks
        if "```json" in s:
            try:
                # Extract JSON between ```json and ``` markers
                s = s.split("```json")[1].split("```")[0].strip()
            except:
                pass
        
        # Attempt to parse the JSON string
        try:
            return json.loads(s)
        except:
            # Log first 200 chars of failed parse for debugging
            logger.debug(f"Failed to parse JSON: {s[:200]}")
            return None

    def _to_detections(
        self, 
        boxes: List[Dict], 
        image_width: int, 
        image_height: int, 
        image_grid_thw: torch.Tensor = None, 
        patch_size: int = 14
    ) -> fo.Detections:
        """
        Convert Kimi's bounding boxes to FiftyOne Detections.
        
        Args:
            boxes: List of dictionaries containing bounding box info
            image_width: Width of the original image in pixels
            image_height: Height of the original image in pixels
            image_grid_thw: Optional tensor with processed image grid dimensions
            patch_size: Size of each patch (default 14)
        
        Returns:
            fo.Detections object with normalized coordinates
        """
        detections = []
        
        # Handle case where boxes is a dictionary - extract list value if present
        if isinstance(boxes, dict):
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
                )
                detections.append(detection)
                
            except Exception as e:
                # Optionally log errors
                print(f"Error processing box {box}: {e}")
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
        """Convert OCR results to FiftyOne Detections.
        
        Args:
            boxes: List of dictionaries containing OCR detection info
            image_width: Width of the original image in pixels
            image_height: Height of the original image in pixels
            image_grid_thw: Optional tensor with processed image grid dimensions
            patch_size: Size of each patch (default 14)
            
        Returns:
            fo.Detections object containing the OCR annotations with text content
        """
        detections = []
        
        # Handle case where boxes is a dictionary - extract list value if present
        if isinstance(boxes, dict):
            boxes = next((v for v in boxes.values() if isinstance(v, list)), boxes)
        
        # Ensure boxes is a list, even for single box input
        boxes = boxes if isinstance(boxes, list) else [boxes]
        
        # Process each OCR box
        for box in boxes:
            try:
                # Extract the bounding box coordinates and text content
                bbox = box.get('bbox')  # [x1,y1,x2,y2] coordinates
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
                )
                detections.append(detection)
                
            except Exception as e:
                # Log any errors processing individual boxes but continue
                logger.debug(f"Error processing OCR box {box}: {e}")
                continue
                
        # Return all detections wrapped in a FiftyOne Detections container
        return fo.Detections(detections=detections)
    
    def _to_keypoints(self, points: List[Dict]) -> fo.Keypoints:
        """Convert a list of point dictionaries to FiftyOne Keypoints.
        
        Args:
            points: List of dictionaries containing point information.
                Each point should have:
                - 'point_2d': List of [x,y] coordinates in [0,1] range
                - 'label': String label describing the point
                
        Returns:
            fo.Keypoints object containing the keypoint annotations
        
        Expected input format:
        [
            {"point_2d": [0.1, 0.2], "label": "person's head", "confidence": 0.9},
            {"point_2d": [0.3, 0.4], "label": "dog's nose"}
        ]
        """
        # Initialize empty list to store converted keypoints
        keypoints = []
        
        # Process each point dictionary from the input list
        for point in points:
            try:
                # Extract the x,y coordinates from the point_2d field (coordinates should already be normalized to [0,1])
                point_2d = point["point_2d"]
                
                # If coordinates are PyTorch tensors, convert them to Python floats
                if torch.is_tensor(point_2d[0]):
                    point_2d = [float(p.cpu()) for p in point_2d]  # Move to CPU and convert to float
                
                # Create a FiftyOne Keypoint object with label and coordinates
                # Use .get() with default "point" label in case label field is missing
                keypoint = fo.Keypoint(
                    label=str(point.get("label", "point")),  # Convert label to string for consistency
                    points=[point_2d],  # Wrap point_2d in list since Keypoint expects list of points
                )
                keypoints.append(keypoint)  # Add the keypoint to our collection
                
            except Exception as e:
                # Log any errors but continue processing remaining points
                logger.debug(f"Error processing point {point}: {e}")
                continue
                
        # Return all keypoints wrapped in a FiftyOne Keypoints container
        return fo.Keypoints(keypoints=keypoints)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert a list of classification dictionaries to FiftyOne Classifications.
        
        Args:
            classes: List of dictionaries containing classification information.
                Each dictionary should have:
                - 'label': String class label
                
        Returns:
            fo.Classifications object containing the converted classification 
            annotations with labels and optional confidence scores
            
        Example input:
            [
                {"label": "cat",},
                {"label": "dog"}
            ]
        """
        classifications = []
        
        # Process each classification dictionary
        for cls in classes:
            try:
                # Create Classification object with required label and optional confidence
                classification = fo.Classification(
                    label=str(cls["label"]),  # Convert label to string for consistency
                )
                classifications.append(classification)
            except Exception as e:
                # Log any errors but continue processing remaining classifications
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        # Return Classifications container with all processed results
        return fo.Classifications(classifications=classifications)


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
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                self.prompt = str(field_value)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"image": sample.filepath if sample else None}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
            )
        
        inputs = self.processor(
            text=[text], 
            images=[image], 
            padding=True, 
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

        # For VQA and agentic, return the raw text output
        if self.operation in ["vqa", "agentic"]:
            return output_text.strip()

        # For other operations, parse JSON and convert to appropriate format
        parsed_output = self._parse_json(output_text)
        if not parsed_output:
            return None

        if self.operation == "detect":
            return self._to_detections(
                parsed_output, 
                image_width=sample.metadata.width, 
                image_height=sample.metadata.height,
                image_grid_thw=inputs['image_grid_hws']
                )
        
        elif self.operation == "point":
            return self._to_keypoints(parsed_output)
        elif self.operation == "classify":
            return self._to_classifications(parsed_output)
        elif self.operation == "ocr":
            return self._to_ocr_detections(
                parsed_output, 
                image_width=sample.metadata.width, 
                image_height=sample.metadata.height,
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
