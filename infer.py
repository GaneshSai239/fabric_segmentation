import os
from PIL import Image
from ultralytics import YOLO

MODEL_DIR = 'best2.pt'
INPUT_FOLDER = 'data/Task4 data'
OUTPUT_FOLDER = './output_images'

def inference_images(image_path, model):
    image = Image.open(image_path)
    predict = model.predict(image)
    boxes = predict[0].boxes
    plotted = predict[0].plot()[:, :, ::-1] 
    if len(boxes) == 0:
        print(f"No detection in {image_path}")
    else:
        output_path = os.path.join(OUTPUT_FOLDER, f"detection_{os.path.basename(image_path)}")
        Image.fromarray(plotted).save(output_path)
        print(f"Detected image saved at {output_path}")
    return predict


if __name__ == '__main__':
    model = YOLO(MODEL_DIR)
    predictsions = inference_images("data\Task4 data\Fabric23.jpg",model)
    print(predictsions)
