"""
RL Threshold Optimizer - CyberGuard AI
========================================

Reinforcement Learning ile adaptif alarm eşiği optimizasyonu.

DQN (Deep Q-Network) kullanarak:
    - False Positive minimize
    - False Negative minimize
    - Dynamic threshold adjustment

State: Traffic stats + Model confidence
Action: Alert / Ignore / Request More
Reward: +1 correct, -10 FP, -100 FN
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import random
from datetime import datetime
import logging
import json

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger("RLThreshold")


class RLState:
    """
    RL State representation

    State features:
    - Model confidence (0-1)
    - Anomaly score (0-1)
    - Historical alert rate
    - Time of day
    - Traffic volume
    - Recent FP/FN rates
    """

    STATE_DIM = 10

    def __init__(self):
        self.features = [
            "model_confidence",
            "anomaly_score",
            "alert_rate_1h",
            "alert_rate_24h",
            "traffic_volume",
            "time_of_day",
            "recent_fp_rate",
            "recent_fn_rate",
            "avg_confidence",
            "confidence_variance",
        ]

    def extract(
        self,
        model_confidence: float,
        anomaly_score: float,
        history: Dict = None,
    ) -> np.ndarray:
        """State vektörü oluştur"""
        history = history or {}

        state = np.zeros(self.STATE_DIM, dtype=np.float32)

        state[0] = model_confidence
        state[1] = anomaly_score
        state[2] = history.get("alert_rate_1h", 0.0)
        state[3] = history.get("alert_rate_24h", 0.0)
        state[4] = min(history.get("traffic_volume", 100) / 1000, 1.0)
        state[5] = history.get("time_of_day", 12) / 24.0
        state[6] = history.get("recent_fp_rate", 0.0)
        state[7] = history.get("recent_fn_rate", 0.0)
        state[8] = history.get("avg_confidence", 0.5)
        state[9] = history.get("confidence_variance", 0.1)

        return state


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class RLThresholdAgent:
    """
    DQN-based Threshold Optimizer

    Actions:
        0: ALERT - Kesin alarm ver
        1: IGNORE - Görmezden gel
        2: REQUEST_MORE - Daha fazla veri iste

    Rewards:
        +1: Correct decision
        -10: False Positive (gereksiz alarm)
        -100: False Negative (kaçırılan saldırı)
        -1: REQUEST_MORE (kaynak maliyeti)
    """

    ACTIONS = ["ALERT", "IGNORE", "REQUEST_MORE"]

    REWARDS = {
        "correct_alert": 1.0,
        "correct_ignore": 1.0,
        "false_positive": -10.0,
        "false_negative": -100.0,
        "request_more": -1.0,
    }

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
    ):
        """
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration
            epsilon_decay: Exploration decay
            buffer_size: Replay buffer size
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow gerekli!")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Models
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self._update_target_network()

        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.state_extractor = RLState()

        # Stats
        self.total_steps = 0
        self.training_episodes = 0
        self.decision_history = []

        logger.info(f"🎮 RLThresholdAgent initialized")
        logger.info(f"   State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"   Epsilon: {epsilon}, Gamma: {gamma}")

    def _build_network(self) -> keras.Model:
        """DQN network oluştur"""
        inputs = layers.Input(shape=(self.state_dim,))

        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)

        outputs = layers.Dense(self.action_dim, activation="linear")(x)

        model = keras.Model(inputs, outputs, name="dqn")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

        return model

    def _update_target_network(self):
        """Target network'ü güncelle"""
        self.target_network.set_weights(self.q_network.get_weights())

    def get_state(
        self,
        model_confidence: float,
        anomaly_score: float,
        history: Dict = None,
    ) -> np.ndarray:
        """State vektörü oluştur"""
        return self.state_extractor.extract(model_confidence, anomaly_score, history)

    def decide(
        self,
        state: np.ndarray,
        training: bool = False,
    ) -> Tuple[int, str]:
        """
        Karar ver

        Args:
            state: State vector
            training: Training modunda epsilon-greedy kullan

        Returns:
            (action_idx, action_name)
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
            action = np.argmax(q_values)

        return action, self.ACTIONS[action]

    def compute_reward(self, action: int, is_attack: bool) -> float:
        """
        Reward hesapla

        Args:
            action: 0=ALERT, 1=IGNORE, 2=REQUEST_MORE
            is_attack: Gerçekten saldırı mı?

        Returns:
            Reward value
        """
        if action == 2:  # REQUEST_MORE
            return self.REWARDS["request_more"]

        if action == 0:  # ALERT
            if is_attack:
                return self.REWARDS["correct_alert"]
            else:
                return self.REWARDS["false_positive"]

        if action == 1:  # IGNORE
            if is_attack:
                return self.REWARDS["false_negative"]
            else:
                return self.REWARDS["correct_ignore"]

        return 0.0

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        """Experience'ı buffer'a ekle"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def train_step(self, batch_size: int = 32) -> float:
        """
        Single training step

        Returns:
            Loss value
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        # Compute targets
        next_q_values = self.target_network.predict(next_states, verbose=0)
        max_next_q = np.max(next_q_values, axis=1)

        targets = rewards + (1 - dones.astype(float)) * self.gamma * max_next_q

        # Current Q values
        current_q = self.q_network.predict(states, verbose=0)

        # Update Q values for taken actions
        for i, action in enumerate(actions):
            current_q[i, action] = targets[i]

        # Train
        loss = self.q_network.train_on_batch(states, current_q)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def train(
        self,
        episodes: int = 100,
        steps_per_episode: int = 100,
        target_update_freq: int = 10,
    ) -> Dict:
        """
        Training loop

        Args:
            episodes: Episode sayısı
            steps_per_episode: Her episode'da adım sayısı
            target_update_freq: Target network güncelleme sıklığı

        Returns:
            Training stats
        """
        logger.info(f"🏋️ Training for {episodes} episodes...")

        total_rewards = []
        losses = []

        for episode in range(episodes):
            episode_reward = 0
            episode_loss = 0

            for step in range(steps_per_episode):
                # Simulated state
                state = np.random.randn(self.state_dim).astype(np.float32)

                # Decide
                action, _ = self.decide(state, training=True)

                # Simulated environment response
                is_attack = random.random() < 0.3  # %30 saldırı olasılığı
                reward = self.compute_reward(action, is_attack)

                # Next state
                next_state = (
                    state + np.random.randn(self.state_dim).astype(np.float32) * 0.1
                )

                # Store experience
                self.store_experience(state, action, reward, next_state, False)

                # Train
                loss = self.train_step()

                episode_reward += reward
                episode_loss += loss

            total_rewards.append(episode_reward)
            losses.append(episode_loss / steps_per_episode)

            # Update target network
            if (episode + 1) % target_update_freq == 0:
                self._update_target_network()

            if (episode + 1) % 10 == 0:
                logger.info(
                    f"  Episode {episode+1}/{episodes}, Reward: {episode_reward:.1f}, Epsilon: {self.epsilon:.3f}"
                )

        self.training_episodes += episodes

        return {
            "episodes": episodes,
            "avg_reward": float(np.mean(total_rewards[-10:])),
            "final_epsilon": self.epsilon,
            "avg_loss": float(np.mean(losses[-10:])),
        }

    def get_threshold_recommendation(
        self,
        model_confidence: float,
        anomaly_score: float,
        history: Dict = None,
    ) -> Dict:
        """
        Threshold önerisi

        Returns:
            {
                "action": str,
                "should_alert": bool,
                "confidence": float,
                "reasoning": str
            }
        """
        state = self.get_state(model_confidence, anomaly_score, history)
        action_idx, action_name = self.decide(state, training=False)

        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
        confidence = float(np.max(q_values) / (np.sum(np.abs(q_values)) + 1e-8))

        # Reasoning
        if action_name == "ALERT":
            reasoning = f"High threat confidence ({model_confidence:.2f}) with anomaly score {anomaly_score:.2f}"
        elif action_name == "IGNORE":
            reasoning = (
                f"Low threat confidence ({model_confidence:.2f}), likely benign traffic"
            )
        else:
            reasoning = f"Borderline confidence ({model_confidence:.2f}), requesting additional analysis"

        return {
            "action": action_name,
            "should_alert": action_name == "ALERT",
            "confidence": confidence,
            "q_values": {a: float(q) for a, q in zip(self.ACTIONS, q_values)},
            "reasoning": reasoning,
            "state_summary": {
                "model_confidence": model_confidence,
                "anomaly_score": anomaly_score,
            },
        }

    def get_stats(self) -> Dict:
        """Agent istatistikleri"""
        return {
            "total_steps": self.total_steps,
            "training_episodes": self.training_episodes,
            "current_epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "actions": self.ACTIONS,
            "rewards": self.REWARDS,
        }

    def save(self, path: str):
        """Agent'ı kaydet"""
        os.makedirs(path, exist_ok=True)

        self.q_network.save(os.path.join(path, "q_network.h5"))
        self.target_network.save(os.path.join(path, "target_network.h5"))

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(
                {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "epsilon": self.epsilon,
                    "total_steps": self.total_steps,
                    "training_episodes": self.training_episodes,
                    "saved_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        logger.info(f"✅ Agent saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RLThresholdAgent":
        """Agent'ı yükle"""
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        agent = cls(
            state_dim=metadata["state_dim"],
            action_dim=metadata["action_dim"],
            epsilon=metadata["epsilon"],
        )

        agent.q_network = keras.models.load_model(os.path.join(path, "q_network.h5"))
        agent.target_network = keras.models.load_model(
            os.path.join(path, "target_network.h5")
        )
        agent.total_steps = metadata["total_steps"]
        agent.training_episodes = metadata["training_episodes"]

        logger.info(f"✅ Agent loaded from {path}")
        return agent


# ============= Factory Functions =============


def create_rl_agent(
    pretrained: bool = False,
) -> RLThresholdAgent:
    """Create RL agent"""
    agent = RLThresholdAgent()

    if pretrained:
        # Quick pretraining
        agent.train(episodes=50, steps_per_episode=50)

    return agent


# ============= Double DQN (Advanced) =============


class DoubleDQNAgent(RLThresholdAgent):
    """
    Double DQN Threshold Optimizer

    Standard DQN'nin overestimation bias sorununu çözer.

    Key difference:
        - Action selection: Online network (Q)
        - Value estimation: Target network (Q')

    Formula:
        y = r + γ * Q'(s', argmax_a Q(s', a))

    Reference:
        "Deep Reinforcement Learning with Double Q-learning" (Van Hasselt et al., 2015)

    Avantajlar:
        - Daha stabil training
        - Daha düşük overestimation
        - Daha iyi policy
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        soft_update_tau: float = 0.005,
    ):
        """
        Args:
            soft_update_tau: Soft update weight (τ) for target network
        """
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
        )
        self.soft_update_tau = soft_update_tau
        logger.info(f"🎮 DoubleDQNAgent initialized with τ={soft_update_tau}")

    def train_step(self, batch_size: int = 32) -> float:
        """
        Double DQN training step

        Key difference from standard DQN:
            - Use online network to SELECT best action for next state
            - Use target network to EVALUATE that action's value
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        # === DOUBLE DQN UPDATE ===

        # Step 1: Use ONLINE network to select best action for next state
        next_q_values_online = self.q_network.predict(next_states, verbose=0)
        best_actions = np.argmax(next_q_values_online, axis=1)

        # Step 2: Use TARGET network to evaluate those actions
        next_q_values_target = self.target_network.predict(next_states, verbose=0)

        # Get Q values for the actions selected by online network
        max_next_q = next_q_values_target[np.arange(batch_size), best_actions]

        # Compute targets
        targets = rewards + (1 - dones.astype(float)) * self.gamma * max_next_q

        # Current Q values
        current_q = self.q_network.predict(states, verbose=0)

        # Update Q values for taken actions
        for i, action in enumerate(actions):
            current_q[i, action] = targets[i]

        # Train
        loss = self.q_network.train_on_batch(states, current_q)

        # Soft update target network (Polyak averaging)
        self._soft_update_target_network()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def _soft_update_target_network(self):
        """
        Soft update target network weights

        θ_target = τ * θ_online + (1 - τ) * θ_target

        More stable than hard updates (periodic copying)
        """
        online_weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()

        new_weights = []
        for online_w, target_w in zip(online_weights, target_weights):
            new_w = (
                self.soft_update_tau * online_w + (1 - self.soft_update_tau) * target_w
            )
            new_weights.append(new_w)

        self.target_network.set_weights(new_weights)

    def get_q_value_stats(self, states: np.ndarray) -> Dict:
        """
        Q-value statistics for diagnosis

        Compare online vs target network to detect overestimation
        """
        online_q = self.q_network.predict(states, verbose=0)
        target_q = self.target_network.predict(states, verbose=0)

        online_max = np.max(online_q, axis=1)
        target_max = np.max(target_q, axis=1)

        overestimation = online_max - target_max

        return {
            "online_q_mean": float(np.mean(online_max)),
            "target_q_mean": float(np.mean(target_max)),
            "overestimation_mean": float(np.mean(overestimation)),
            "overestimation_std": float(np.std(overestimation)),
            "online_q_std": float(np.std(online_max)),
        }


def create_double_dqn_agent(
    pretrained: bool = False,
    tau: float = 0.005,
) -> DoubleDQNAgent:
    """
    Factory function for Double DQN agent

    Args:
        pretrained: Whether to pretrain the agent
        tau: Soft update weight (higher = faster adaptation)

    Recommended tau values:
        - τ=0.001: Very slow, very stable
        - τ=0.005: Balanced (default)
        - τ=0.01: Faster adaptation, less stable
    """
    agent = DoubleDQNAgent(soft_update_tau=tau)

    if pretrained:
        agent.train(episodes=50, steps_per_episode=50)

    return agent


# ============= Test =============

if __name__ == "__main__":
    print("🧪 RL Threshold Agent Test\n")

    # Create agent
    agent = RLThresholdAgent()

    # Quick training
    print("Training DQN...")
    result = agent.train(episodes=20, steps_per_episode=50)
    print(f"DQN Training result: {result}")

    # Double DQN Test
    print("\n🔥 Double DQN Test")
    double_agent = DoubleDQNAgent(soft_update_tau=0.005)
    double_result = double_agent.train(episodes=20, steps_per_episode=50)
    print(f"Double DQN Training result: {double_result}")

    # Q-value comparison
    test_states = np.random.randn(10, 10).astype(np.float32)
    q_stats = double_agent.get_q_value_stats(test_states)
    print(f"Q-value overestimation: {q_stats['overestimation_mean']:.4f}")

    # Test decision
    print("\n📊 Test Decisions:")

    test_cases = [
        (0.9, 0.8, "High confidence, high anomaly"),
        (0.3, 0.2, "Low confidence, low anomaly"),
        (0.5, 0.5, "Medium confidence, medium anomaly"),
    ]

    for confidence, anomaly, desc in test_cases:
        rec = agent.get_threshold_recommendation(confidence, anomaly)
        print(f"\n{desc}:")
        print(f"  Action: {rec['action']}")
        print(f"  Should Alert: {rec['should_alert']}")
        print(f"  Reasoning: {rec['reasoning']}")

    print("\n✅ Test completed!")
