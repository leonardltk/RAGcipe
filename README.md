# Init
```bash
pip install requirements.txt
```

# mmocr
```bash
git clone git@github.com:open-mmlab/mmocr.git

cd mmocr
python tools/infer.py demo/demo_text_ocr.jpg --det DBNet --rec CRNN --show --print-result

# conda activate demo_v4
# export PATH="/c/ffmpeg/bin:$PATH"
# export KMP_DUPLICATE_LIB_OK=TRUE
```

# exp
```bash
python run.py
```

# image
![gradio interface](images/interface.png)
