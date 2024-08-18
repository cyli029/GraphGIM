import sys

sys.path.append('..')
from model.teacher import ResumeSingleTeacher
from PIL import Image
from torchvision import transforms, models
from torchvision.utils import make_grid
import numpy as np
import torch
import torch.nn as nn


model_name = 'resnet18'
resume_teacher = '../resumes/pretrained-teachers/' + 'IEM' + '.pth'
resume_teacher_name = 'image3d_teacher'
image_encoder_teacher = ResumeSingleTeacher(model_name, resume_teacher, resume_teacher_name)
image3d_path = '../dataset/pre-training/iem-200w/processed/image3d/0/0.png'
image3d = Image.open(image3d_path)

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
image = transform(image3d).unsqueeze(0)


def save_image(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')


model = image_encoder_teacher.teacher.model
model.eval()
layer1_model = nn.Sequential(*list(model.children())[:5])
f3 = layer1_model(image)
save_image(f3,'layer1')

new_model = nn.Sequential(*list(model.children())[:6])
f4 = new_model(image)  # [1, 128, 28, 28]
save_image(f4, 'layer2')

new_model = nn.Sequential(*list(model.children())[:7])
f5 = new_model(image)  # [1, 128, 28, 28]
save_image(f5, 'layer3')


new_model = nn.Sequential(*list(model.children())[:8])
f6 = new_model(image)  # [1, 128, 28, 28]
save_image(f5, 'layer4')