import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


# Дефиниция нейронной сети для DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Агент DQN
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space.n
        self.state_space = env.observation_space.shape[0]

        # Модель нейронной сети
        self.model = DQN(self.state_space, self.action_space)
        self.target_model = DQN(self.state_space, self.action_space)
        self.target_model.load_state_dict(self.model.state_dict())

        # Оптимизатор и критерий
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Опытный буфер
        self.memory = deque(maxlen=10000)

        # Гиперпараметры
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_target_every = 10

    # Выбор действия с использованием epsilon-greedy
    def act(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()  # случайное действие
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    # Заполнение буфера
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Обучение на случайном батче
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack([torch.FloatTensor(s) for s in states]).float()
        next_states = torch.stack([torch.FloatTensor(s) for s in next_states]).float()
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()

        # Предсказания и целевые значения
        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]

        # Целевая функция
        target = rewards + (self.gamma * next_q_value * (1 - dones))

        # Потери и обновление модели
        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Обновление epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Обновление целевой модели
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


# Обучение DQN
def train_dqn(agent, env, n_episodes=1000):
    scores = []
    for e in range(n_episodes):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            score += reward

        scores.append(score)

        # Обновление целевой модели
        if e % agent.update_target_every == 0:
            agent.update_target()

        print(f"Episode {e + 1}, Total Reward: {score}, Epsilon: {agent.epsilon:.4f}")

    return scores


if __name__ == "__main__":
    env = gym.make("LunarLander-v3", render_mode="human")  # Для использования с графическим интерфейсом
    agent = DQNAgent(env)
    n_episodes = 500  # Количество эпизодов

    scores = train_dqn(agent, env, n_episodes=n_episodes)
    print("Обучение завершено!")
