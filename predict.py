from torchvision import transforms
from PIL import Image
import torch
from models.model import SoundCNN

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
          'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SoundCNN(num_classes=len(labels))
model.load_state_dict(torch.load('sound_cnn.pth', map_location=device))
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    pred = output.argmax(1).item()
    return labels[pred]
