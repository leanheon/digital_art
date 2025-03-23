import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import time

# Load pre-trained DeepLabv3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define preprocessing transform
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_silhouette(frame):
    # Preprocess the frame
    input_tensor = preprocess(frame).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Get the person class (15 in COCO dataset)
    person_mask = (output.argmax(0) == 15).cpu().numpy().astype(np.uint8) * 255
    
    return person_mask

def real_time_silhouette_extraction():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    prev_frame_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract silhouette
        silhouette = extract_silhouette(frame)
        
        # Apply silhouette mask to original frame
        result = cv2.bitwise_and(frame, frame, mask=silhouette)
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {fps:.2f}"
        
        # Display FPS on frame
        cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display result
        cv2.imshow('Real-time Human Silhouette', result)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_silhouette_extraction()