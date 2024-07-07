import cv2
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import numpy as np

# Load the DeepLabv3 model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

# Define the transformation
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a VideoCapture object
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = transform(frame).unsqueeze(0)
    
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Get the prediction
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Create a mask for the human class (class 15 in COCO dataset)
    mask = (output_predictions == 15).astype(np.uint8) * 255

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the result
    cv2.imshow('Human Silhouette', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()