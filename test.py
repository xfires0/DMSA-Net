import warnings
warnings.filterwarnings('ignore')
from ultralytics import DMSA

if __name__ == '__main__':
    model = DMSA('best.pt')
    model.predict(source=r'C:\Users\MSI\test_features',
                  imgsz=800,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  visualize=True
                )