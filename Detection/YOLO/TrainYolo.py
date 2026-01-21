from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    model = YOLO("yolo11s.pt")
    data_path=   Path(__file__).resolve().parent.parent.parent.parent/ "data"/"yolo_dataset"/"data.yaml" 


    model.train(
        pretrained=True,
        data=data_path,
        epochs=250,
        fraction=0.5,    # 50% of the dataset for fast testing
        batch=0.7,
        seed = 42,
        patience = 20,
        project="./yolo-runs"
    )
