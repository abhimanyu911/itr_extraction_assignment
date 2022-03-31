# ITR Extractor
A streamlit-aided web app for Income Tax Return field extraction. 

Pipeline:
Image annotation -> Object Detection -> Extract ROI -> OCR -> Requisite text

# How to get started ?

1. Refer [custom training tutorial](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for yolov5 - it is recommending to use one of their ready-made environments for training in order to avoid dependency issues 
2. Refer [train_val_test.ipynb]() to observe how I trained, validated and tested my model on google colab using google drive as a storage option
3. After training, validating and testing, downloading the 'best.pt' best weights from the directory specified by yolo, clone this repo to a local directory and place the best.pt file **alongside** the rest of the contents in this directory and rename to 'best_weights.pt'

# Installations
1. Install tesseract OCR engine from [here](https://github.com/UB-Mannheim/tesseract/wiki)
2. Place appropriate location for tesseract.exe file in the [test_ocr.py script]()
3. Run:

```
pip install -r requirements.txt
```

# Run Extractor
To run the web app, perform the following command in the terminal:

```
cd path/to/directory
streamlit run app.py
```

# Object detection results

| Image set               | mAP @ 0.5     | mAP @ 0.5:0.95     |
| ----------------------- | ------------- | ------------------ |
| Validation(all classes) | 96.9          | 77.2               |
| Testing(all classes)    | 92.1          | 68.3               |