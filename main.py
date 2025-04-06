"""
Author: Samuel Augusto
Date: 06/04/2025
Description: ChatBot IA
Instalation:
Digite isso no terminal para rodar o arquivo
pip install nltk
pip install torch numpy
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
python main.py
"""
# Importando módulos necessários:
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.stem import WordNetLemmatizer
# Definindo a estrutura para a rede neural herdando de nn.Module a base para o modelo PyTorch
class NeuralNetwork(nn.Module):
    # Inicialização da rede neural
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # input_size: número de entradas das palavras
        self.l2 = nn.Linear(hidden_size, hidden_size) # hidden_size: número de neuronios na camada oculta
        self.l3 = nn.Linear(hidden_size, output_size) # output_size: número de possiveis respostas
        self.relu = nn.ReLU() # Função que corrige valores negativos para zero
    # Passagem dos dados pela rede neural
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # Sem softmax pois já está incluído no CrossEntropyLoss
        return out
# Gerenciamento de dados, textos, treinamento e resposta
class ChatbotAI:
    def __init__(self, intents_file='intents.json'):
        self.intents_file = intents_file # Caminho do json para as intents
        self.lemmatizer = WordNetLemmatizer() # Reduz palavras a forma raiz
        self.intents = None
        # Armazena palavras e categorias
        self.tags = []
        self.all_words = []
        self.xy = []
        # Armazenar os modelos e a inicialização dos dados da rede neural
        self.model = None
        self.input_size = 0
        self.hidden_size = 8
        self.output_size = 0
    # Carrega o conteudo do intents com os padrões
    def load_intents(self):
        with open(self.intents_file, 'r') as f:
            self.intents = json.load(f)
    # Le o padrão e faz a remoção de pontuação
    def preprocess_data(self):
        # Extrair todas as palavras e tags
        for intent in self.intents['intents']:
            tag = intent['tag']
            self.tags.append(tag)
            for pattern in intent['patterns']:
                # Tokenizar cada palavra
                words = nltk.word_tokenize(pattern)
                # Adicionar a lista de palavras
                self.all_words.extend(words)
                # Adicionar ao par (palavras, tag)
                self.xy.append((words, tag))
        # Lematizar, converter para minúsculas e remover pontuação
        ignore_chars = ['?', '!', '.', ',']
        self.all_words = [self.lemmatizer.lemmatize(word.lower()) for word in self.all_words if word not in ignore_chars]
        # Remover duplicatas e ordenar
        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))
    # Converte as frases em vetores e associa cada um com sua tag
    def create_training_data(self):
        # Criar dados de treinamento
        X_train = []
        y_train = []
        for (pattern_words, tag) in self.xy:
            # Criar bag of words para cada pattern
            bag = self._bag_of_words(pattern_words)
            X_train.append(bag)
            # Label: tag correspondente
            label = self.tags.index(tag)
            y_train.append(label)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        return X_train, y_train
    # Cria os vetores de armazenamento de palavras -> [0,0,1,1,0,1,1,0]
    def _bag_of_words(self, tokenized_sentence):
        # Lematizar palavras
        tokenized_sentence = [self.lemmatizer.lemmatize(word.lower()) for word in tokenized_sentence]
        # Criar bag of words
        bag = [0] * len(self.all_words)
        for idx, word in enumerate(self.all_words):
            if word in tokenized_sentence:
                bag[idx] = 1
        return np.array(bag)
    # Cria um dataset personalizado e define o modelo e o treinamento respectivo
    def train_model(self, epochs=1000, batch_size=8, learning_rate=0.001, print_every=100):
        # Preparar dados de treinamento
        X_train, y_train = self.create_training_data()
        # Hiperparâmetros
        self.input_size = len(X_train[0])
        self.output_size = len(self.tags)
        # Criar dataset
        class ChatDataset(Dataset):
            def __init__(self, X_data, y_data):
                self.n_samples = len(X_data)
                self.x_data = X_train
                self.y_data = y_train
            def __getitem__(self, index):
                return self.x_data[index], self.y_data[index]
            def __len__(self):
                return self.n_samples
        dataset = ChatDataset(X_train, y_train)
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        # Configurar dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Criar modelo
        self.model = NeuralNetwork(self.input_size, self.hidden_size, self.output_size).to(device)
        # Definir função de perda e otimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Treinar o modelo
        for epoch in range(epochs):
            for (words, labels) in train_loader:
                words = words.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                # Forward
                outputs = self.model(words)
                loss = criterion(outputs, labels)
                # Backward e otimização
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1)%print_every==0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        print(f'Treinamento finalizado. Perda final: {loss.item():.4f}')
        # Salvando os dados do modelo
        model_data = {
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "all_words": self.all_words,
            "tags": self.tags
        }
        torch.save(model_data, "model_data.pth")
        print('Modelo salvo com sucesso!')
    # Carregando os dados do modelo assim como abrindo o arquivo intents
    def load_model(self, model_file="model_data.pth"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.load(model_file)
        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.output_size = data["output_size"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]
        self.model = NeuralNetwork(self.input_size, self.hidden_size, self.output_size).to(device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
        with open(self.intents_file, 'r') as f:
            self.intents = json.load(f)
    def get_response(self, sentence):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Tokenizar a sentença
        sentence = nltk.word_tokenize(sentence)
        # Criar bag of words
        X = self._bag_of_words(sentence)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device, dtype=torch.float)
        # Fazer a predição
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]
        # Verificar probabilidade
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        # Se a probabilidade for maior que um threshold, retornar resposta
        if prob.item() > 0.75:
            for intent in self.intents['intents']:
                if intent['tag'] == tag:
                    # Verificar se precisamos chamar uma função específica
                    if tag == "stocks":
                        # Simular uma função de ações
                        stocks = ['APPL', 'META', 'NVDA', 'GS', 'MSFT']
                        return f"Aqui estão suas ações: {', '.join(random.sample(stocks, 3))}"
                    else:
                        return random.choice(intent['responses'])
        else:
            return "Desculpe, não entendi o que você quis dizer."
# Função para criar o arquivo intents.json
def create_intents_file():
    intents = {
        "intents": [
            {
            "tag": "greeting",
            "patterns": ["Oi", "Como vai", "Tem alguém ai?", "Ola", "Bom dia", "Eai", "Oi tudo bem", "Opa"],
            "responses": ["Ola!", "Otimo ver voce novamente!", "Oi, como posso ajudar?"]
            },
            {
                "tag": "goodbye",
                "patterns": ["tchau", "ate mais", "adeus", "estou saindo", "tenha um bom dia", "ate", "ate logo"],
                "responses": ["Tchau!", "Falamos mais tarde", "Ate logo!"]
            },
            {
                "tag": "programming",
                "patterns": ["O que e programaçao?", "O que é codificaçao?", "Fale sobre programaçao", "Fale sobre codificação", "O que e desenvolvimento de software?"],
                "responses": ["Programaçao ou desenvolvimento de software significa escrever codigo de computador para automatizar tarefas."]
            },
            {
                "tag": "resource",
                "patterns": ["Onde posso aprender a programar?", "Melhor maneira de aprender a programar", "Como posso aprender programaçao", "Bons recursos para programaçao"],
                "responses": ["Existem muitos recursos excelentes como canais do YouTube, cursos onlines e varios websites como o W3schools!"]
            },
            {
                "tag": "name",
                "patterns": ["qual seu nome?", "como voce se chama?", "quem e voce?", "como posso te chamar?"],
                "responses": ["Eu sou IABot, seu assistente de IA!", "Me chamo IABot, em que posso ajudar?", "IABot ao seu dispor!"]
            },
            {
                "tag": "samuel",
                "patterns": ["quem e Samuel Augusto?", "O que ele faz?", "Mostre o portfolio de Samuel"],
                "responses": ["Samuel Augusto e um excelente programador que busca conhecimento e aprender", "O portfolio dele é: https://isamuelaugustoi.github.io/Portfolio-SA/"]
            },
            {
                "tag": "help",
                "patterns": ["voce pode me ajudar?", "preciso de ajuda", "me ajude", "o que você faz?"],
                "responses": ["Claro! Posso ajudar com informaçoes sobre programaçao, recomendar recursos de aprendizado, falar sobre o meu criador e muito mais."]
            }
        ]
    }
    with open("intents.json", "w", encoding="utf-8") as f:
        json.dump(intents, f, ensure_ascii=False, indent=4)
    
    print("Arquivo intents.json criado com sucesso!")

# Script principal de entrada
def main():
    # Verificar se o arquivo intents.json existe
    if not os.path.exists("intents.json"):
        print("Arquivo intents.json não encontrado. Criando...")
        create_intents_file()
    # Verificar se o modelo já foi treinado
    if os.path.exists("model_data.pth"):
        print("Modelo encontrado. Carregando...")
        chatbot = ChatbotAI()
        chatbot.load_model()
    else:
        print("Modelo não encontrado. Treinando...")
        chatbot = ChatbotAI()
        chatbot.load_intents()
        chatbot.preprocess_data()
        chatbot.train_model(epochs=1000, print_every=100)
    
    # Interface do chatbot
    print("\n===== IABot - Seu Chatbot de IA =====")
    print("Digite 'sair' para encerrar a conversa.\n")
    while True:
        sentence = input("Você: ")
        if sentence.lower() == 'sair':
            break
        response = chatbot.get_response(sentence)
        print(f"IABot: {response}")
if __name__ == "__main__":
    main()