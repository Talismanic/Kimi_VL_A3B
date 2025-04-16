# Kimi-VL-A3B FiftyOne Zoo Model

This repository provides a FiftyOne Zoo Model integration for the Kimi-VL-A3B multimodal model, enabling various vision-language tasks through a simple interface.

## Features

The model supports multiple operations:

- **Visual Question Answering (VQA)**: Get natural language answers to questions about images
- **Object Detection**: Locate and identify objects, UI elements, and regions of interest
- **OCR**: Detect and extract text from images with bounding boxes and text type classification
- **Keypoint Detection**: Identify precise locations of specific points of interest
- **Classification**: Generate multiple relevant classifications for image content
- **Agentic**: Generate PyAutoGUI code for UI automation based on visual content

## Important Notes and Limitations

### Object Detection Functionality

This model is not designed or optimized for traditional object detection tasks - that is, identifying and localizing objects within images by outputting bounding boxes and class labels. 

While the model can engage in visual reasoning and answer questions about image content, it does not natively output bounding boxes or segmentation masks for objects. There is no evidence in the technical documentation, model cards, or reports of the model supporting standard object detection benchmarks or providing APIs specifically for returning precise object locations.

Your results may vary significantly when using this model for tasks requiring accurate bounding box predictions.

### Forward Compatibility

The object detection functionality has been included in this FiftyOne Zoo implementation as a forward-looking feature. As future versions of the Kimi-VL model potentially add support for precise object detection and localization, this integration will make it seamless to upgrade without requiring significant changes to the codebase or API. This architectural decision allows for easy updates when enhanced detection capabilities become available in future model releases.


## Installation

1. Register the zoo model source:
```python
import fiftyone.zoo as foz
foz.register_zoo_model_source("https://github.com/harpreetsahota204/Kimi_VL_A3B", overwrite=True)
```

2. Download the model (choose one):
```python
# For the Instruct model
foz.download_zoo_model(
    "https://github.com/harpreetsahota204/Kimi_VL_A3B",
    model_name="moonshotai/Kimi-VL-A3B-Instruct"
)

# Or for the Thinking model
foz.download_zoo_model(
    "https://github.com/harpreetsahota204/Kimi_VL_A3B",
    model_name="moonshotai/Kimi-VL-A3B-Thinking"
)
```

3. Load the model:
```python
model = foz.load_zoo_model(
   "moonshotai/Kimi-VL-A3B-Instruct",
   # install_requirements=True #if you are using for the first time and need to download reuirement,
   # ensure_requirements=True #  ensure any requirements are installed before loading the model
   )
```

## Usage

### Visual Question Answering
```python
model.operation = "vqa"
model.prompt = "Describe this screenshot and what the user might be doing in it."
dataset.apply_model(model, label_field="vqa_results")
```

### Object Detection
```python
model.operation = "detect"
model.prompt = "Locate the elements of this UI that a user can interact with."
dataset.apply_model(model, label_field="detections")
```

### OCR
```python
model.operation = "ocr"
model.prompt = "Read all the text in the image."
dataset.apply_model(model, label_field="ocr_results")
```

### Keypoint Detection
```python
model.operation = "point"
model.prompt = "Point to all the interactive elements in UI."
dataset.apply_model(model, label_field="keypoints")
```

### Classification
```python
model.operation = "classify"
model.prompt = "List the type of operating system, open application, and what the user is working on."
dataset.apply_model(model, label_field="classifications")
```

### Agentic (UI Automation)
```python
model.operation = "agentic"
model.prompt = "Write code to close application windows and quit the application."
dataset.apply_model(model, label_field="automation_code")
```

### Using Sample Fields as Input
You can also use fields from your dataset samples as prompts:
```python
dataset.apply_model(model, label_field="results", prompt_field="instruction")
```

## Model Details

- The model automatically selects the best available device (CUDA, MPS, or CPU)
- For CUDA and mps devices, it uses `bfloat16` precision for optimal performance
- Temperature settings:
  - Kimi-VL-A3B-Instruct: 0.2
  - Kimi-VL-A3B-Thinking: 0.6


## License

The model is released under MIT License

## Citation

```bibtex
@misc{kimiteam2025kimivltechnicalreport,
      title={{Kimi-VL} Technical Report}, 
      author={Kimi Team and Angang Du and Bohong Yin and Bowei Xing and Bowen Qu and Bowen Wang and Cheng Chen and Chenlin Zhang and Chenzhuang Du and Chu Wei and Congcong Wang and Dehao Zhang and Dikang Du and Dongliang Wang and Enming Yuan and Enzhe Lu and Fang Li and Flood Sung and Guangda Wei and Guokun Lai and Han Zhu and Hao Ding and Hao Hu and Hao Yang and Hao Zhang and Haoning Wu and Haotian Yao and Haoyu Lu and Heng Wang and Hongcheng Gao and Huabin Zheng and Jiaming Li and Jianlin Su and Jianzhou Wang and Jiaqi Deng and Jiezhong Qiu and Jin Xie and Jinhong Wang and Jingyuan Liu and Junjie Yan and Kun Ouyang and Liang Chen and Lin Sui and Longhui Yu and Mengfan Dong and Mengnan Dong and Nuo Xu and Pengyu Cheng and Qizheng Gu and Runjie Zhou and Shaowei Liu and Sihan Cao and Tao Yu and Tianhui Song and Tongtong Bai and Wei Song and Weiran He and Weixiao Huang and Weixin Xu and Xiaokun Yuan and Xingcheng Yao and Xingzhe Wu and Xinxing Zu and Xinyu Zhou and Xinyuan Wang and Y. Charles and Yan Zhong and Yang Li and Yangyang Hu and Yanru Chen and Yejie Wang and Yibo Liu and Yibo Miao and Yidao Qin and Yimin Chen and Yiping Bao and Yiqin Wang and Yongsheng Kang and Yuanxin Liu and Yulun Du and Yuxin Wu and Yuzhi Wang and Yuzi Yan and Zaida Zhou and Zhaowei Li and Zhejun Jiang and Zheng Zhang and Zhilin Yang and Zhiqi Huang and Zihao Huang and Zijia Zhao and Ziwei Chen},
      year={2025},
      eprint={2504.07491},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.07491}, 
}
```