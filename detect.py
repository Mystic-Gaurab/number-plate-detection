from multiprocessing import freeze_support

def main():
    # Your training logic here
    # For example:
    from ultralytics import YOLO
    model = YOLO('yolov8n.yaml')
    model.train(data='data.yaml', epochs=10)

if __name__ == '__main__':
    freeze_support()  # Optional but recommended on Windows
    main()
