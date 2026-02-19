# 🧠 TinyML na SBC Labrador: Classificação com Fashion MNIST

## 📖 Sobre o Projeto
Este repositório contém os códigos e arquivos desenvolvidos para a **Tarefa de Aprendizagem de Máquina - Parte 2**, parte da residência tecnológica **Embarcatech**. O objetivo principal é desenvolver um sistema inteligente embarcado capaz de realizar a classificação local de imagens utilizando uma Rede Neural Convolucional (CNN). O modelo foi otimizado utilizando **Full Integer Quantization (INT8)**, garantindo eficiência, baixo consumo de memória e estabilidade no ambiente embarcado.

## 🛠️ Tecnologias e Hardware
* **Google Colab:** Treinamento e avaliação do modelo original (`.keras`).
* **TensorFlow Lite:** Conversão e quantização do modelo (`.tflite`).
* **SBC Labrador (Caninos Loucos):** Placa utilizada para a inferência local.
* **Python 3:** Linguagem base para os scripts.
* **Bibliotecas:** `tflite-runtime`, `numpy`, `pillow`.

## 📊 Dataset Utilizado
Para atender aos requisitos do projeto, utilizamos o dataset **Fashion MNIST**, que consiste em 70.000 imagens em tons de cinza (28x28 pixels) divididas em 10 categorias de peças de vestuário (camisetas, calças, casacos, tênis, etc.).

## 🚀 Como Executar o Projeto na Labrador

### 1. Preparando o Ambiente
Acesse o terminal da sua SBC Labrador, crie um ambiente virtual Python isolado e instale as dependências:

```bash
# Criar e ativar o ambiente virtual
python3 -m venv myenv
source myenv/bin/activate

# Atualizar o gerenciador de pacotes e instalar bibliotecas
pip install --upgrade pip
pip install numpy pillow tflite-runtime
