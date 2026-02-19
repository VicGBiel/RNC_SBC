import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image 

# 1. Carregar o modelo e alocar tensores
model_path = "model.tflite" # Nome do seu arquivo gerado no Colab
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 2. Obter detalhes de entrada e saída 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Preparar dados de entrada
image_path = "teste_roupa.png" # Imagem que você baixou do Colab
img = Image.open(image_path).convert('L') # 'L' converte para preto e branco (28x28x1)
img = img.resize((28, 28)) # Garante o tamanho certo

# Transformar em array e normalizar (0 a 1), igual fizemos no Colab
input_data = np.array(img, dtype=np.float32)
scale, zero_point = input_details[0]['quantization']
input_data = (input_data / 255.0) / scale + zero_point
input_data = input_data.astype(np.int8) # Força ser inteiro

# Adicionar as dimensões extras (Batch e Canal)
# O shape final deve ser (1, 28, 28, 1)
input_data = np.expand_dims(input_data, axis=0)
input_data = np.expand_dims(input_data, axis=-1)

# 4. Definir a entrada e executar a inferência
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 5. Saída
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_index = np.argmax(output_data)

out_scale, out_zero_point = output_details[0]['quantization']
confidence = (output_data[0][predicted_index] - out_zero_point) * out_scale

# Lista de nomes para facilitar a leitura (Fashion MNIST)
class_names = [
    'Camiseta/Top', 'Calça', 'Pullover', 'Vestido', 'Casaco',
    'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota'
]

print("-" * 30)
print(f"Resultado Bruto: {output_data}")
print("-" * 30)
print(f"IDENTIFICAÇÃO: {class_names[predicted_index]}")
print(f"Confiança: {confidence:.2f}")
print("-" * 30)