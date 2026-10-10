import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Imports do seu projeto
from gwproject.gravity_spy import build_resnet18_grayscale, resolve_device
from gwproject.plotting import FIGURE_DPI

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks para capturar as ativações e os gradientes da última camada
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        # 1. Passa a imagem pela rede
        model_output = self.model(input_tensor)
        
        if class_idx is None:
            # Pega a classe que a IA teve mais certeza
            class_idx = model_output.argmax(dim=1).item()
            
        # 2. Zera os gradientes anteriores
        self.model.zero_grad()
        
        # 3. Faz o "backward" (pega o sinal da classe e volta de ré pela rede)
        target = model_output[0][class_idx]
        target.backward()

        # 4. Calcula os pesos do Grad-CAM (média global dos gradientes)
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        # 5. Multiplica as ativações pelos pesos
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # 6. Aplica ReLU (ignora pixels que votaram "contra" a classe) e normaliza
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam, class_idx


def main():
    parser = argparse.ArgumentParser(description="Script para gerar Grad-CAM em imagens do LIGO")
    parser.add_argument("--image", type=str, required=True, help="Caminho para uma imagem JPG do Gravity Spy")
    parser.add_argument("--model-path", type=str, default=r"outputs\gravity_spy_harmonized\seed_42_cuda_balanced\best_model.pt", help="Caminho do modelo treinado")
    parser.add_argument("--output-dir", type=str, default=None, help="Pasta de saida (padrao: pasta atual)")
    args = parser.parse_args()

    device = resolve_device("auto")
    print(f"Usando processamento: {device}")

    # Carregando o modelo salvo
    print(f"Carregando modelo de: {args.model_path}")
    saved = torch.load(args.model_path, map_location=device, weights_only=False)
    classes = saved["classes"]
    
    model = build_resnet18_grayscale(
        num_classes=len(classes),
        pretrained=False,
        conv1_stride=saved["profile"].get("conv1_stride", 2),
        use_maxpool=saved["profile"].get("use_maxpool", False)
    )
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    model.eval()

    # Selecionando a última camada convolucional da ResNet-18
    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    # Preparando a imagem (mesmo transform do treino)
    raw_image = Image.open(args.image).convert("L")
    transform = transforms.Compose([
        transforms.Resize((140, 170)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # Gerando o Mapa de Calor
    print("Calculando Grad-CAM...")
    heatmap, predicted_class_idx = grad_cam.generate_heatmap(input_tensor)
    predicted_class_name = classes[predicted_class_idx]
    print(f"IA previu a classe: {predicted_class_name}")

    # Redimensionando o heatmap para o tamanho original da imagem
    import cv2
    raw_img_np = np.array(raw_image)
    heatmap_resized = cv2.resize(heatmap, (raw_img_np.shape[1], raw_img_np.shape[0]))
    # applyColorMap devolve BGR; o matplotlib exibe RGB. Sem a conversao, vermelho
    # (atencao maxima) aparece como azul e vice-versa.
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Criando imagem sobreposta (50% heatmap, 50% imagem original em RGB)
    raw_img_rgb = cv2.cvtColor(raw_img_np, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(raw_img_rgb, 0.6, heatmap_colored, 0.4, 0)

    # Salvando os resultados
    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"gradcam_{Path(args.image).stem}__{predicted_class_name}.png"
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_image, cmap='gray')
    axes[0].set_title("1. Imagem Original")
    axes[0].axis('off')

    axes[1].imshow(heatmap_colored)
    axes[1].set_title("2. O Foco da IA (Heatmap)")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f"3. Resultado (Previsão: {predicted_class_name})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=FIGURE_DPI)
    print(f"Sucesso! Imagem final salva como '{output_filename}' .")


if __name__ == "__main__":
    main()
