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
```

### 2. Organizando os Arquivos
Certifique-se de que os seguintes arquivos foram transferidos para o mesmo diretório na placa:
* `model.tflite`: O modelo de rede neural já quantizado para INT8.
* `teste_roupa.png`: Uma imagem de teste gerada a partir do dataset original.
* `inferencia.py`: O script de execução principal.

### 3. Rodando a Inferência
Com o ambiente virtual ativado (`myenv`), execute o script. Ele irá carregar a imagem, aplicar as escalas de quantização inversas e exibir a predição no terminal:

```bash
python3 inferencia.py
```

## 📝 Resultados
A saída no terminal demonstrará a classe predita pelo modelo (ex: "Casaco", "Pullover") juntamente com a sua respectiva porcentagem de confiança matemática, processada inteiramente pela CPU da SBC Labrador.

---
