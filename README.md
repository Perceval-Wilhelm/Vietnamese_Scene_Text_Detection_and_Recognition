# Vietnamese Scene Text Detection and Recognition

This project focuses on detecting and recognizing Vietnamese text in images, leveraging PaddleOCR for model training, evaluation, and inference. For more detailed information, refer to the [blog post](http://tutorials.aiclub.cs.uit.edu.vn/index.php/2022/04/20/nhan-dang-chu-tieng-viet-trong-anh-ngoai-canh/).

## Setup
To set up the environment and install dependencies, follow these steps:

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Train Models
### Train Detection Model
To train the text detection model, use the following command:
```bash
python3 tools/train.py -c ./configs/det/SAST.yml
```

### Train Recognition Model
To train the text recognition model, use the following command:
```bash
python3 tools/train.py -c ./configs/rec/SRN.yml
```

## Evaluation
### Evaluate Detection Model
To evaluate the text detection model, use the following command:
```bash
python3 tools/eval.py -c ./configs/det/SAST.yml
```

### Evaluate Recognition Model
To evaluate the text recognition model, use the following command:
```bash
python3 tools/eval.py -c ./configs/rec/SRN.yml
```

## Prediction
### Predict Detection
To predict using the text detection model, use the following command:
```bash
python3 tools/infer_det.py -c ./configs/det/SAST.yml -o Global.infer_img=#path_to_image
```

### Predict Recognition
To predict using the text recognition model, use the following command:
```bash
python3 tools/infer_rec.py -c ./configs/rec/SRN.yml -o Global.infer_img=im0001_1.jpg
```

## Convert to Inference Model
To convert the trained models to inference models, use the following commands:
```bash
python3 tools/export_model.py -c ./configs/det/SAST.yml  
python3 tools/export_model.py -c ./configs/rec/SRN.yml
```

## Detection and Recognition Concatenation
To run detection and recognition together, use the following command:
```bash
python3 tools/infer/predict_system.py \
    --use_gpu=True \
    --det_algorithm="SAST" \
    --det_model_dir="./inference/SAST" \
    --rec_algorithm="SRN" \
    --rec_model_dir="./inference/SRN/" \
    --rec_image_shape="1, 64, 256" \
    --image_dir=#path_img \
    --rec_char_type="ch" \
    --drop_score=0.7 \
    --rec_char_dict_path="./ppocr/utils/dict/vi_vietnam.txt"
```

## Docker Setup
### Build Docker Image
To build the Docker image, use the following command:
```bash
docker build -t vietnamese-ocr-app .
```

### Run Docker Container
To run the Docker container, use the following command:
```bash
docker run -p 8501:8501 vietnamese-ocr-app
```

## Streamlit Application for Text Detection and Recognition
This Streamlit app allows you to upload an image, run text detection and recognition, and visualize the results. The recognized text will be displayed in its original position within the image.

### Dockerfile
Use the following Dockerfile to set up the Streamlit application:

```dockerfile
# Use the PaddlePaddle Docker image
FROM paddlepaddle/paddle:2.1.2

# Set the working directory in the container
WORKDIR /app

# Install the necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit when the container launches
ENTRYPOINT ["streamlit", "run"]

# Streamlit app filename
CMD ["app.py"]
```

### Running the Streamlit App
Once the Docker container is running, access the Streamlit app in your browser at `http://localhost:8501`.

Upload an image, and the app will display the image with detected text boxes. The recognized text will be shown in the same positions as in the original image.

For any issues or further assistance, please refer to the [blog post](http://tutorials.aiclub.cs.uit.edu.vn/index.php/2022/04/20/nhan-dang-chu-tieng-viet-trong-anh-ngoai-canh/) or contact the project maintainers.
