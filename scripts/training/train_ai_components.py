"""
VAE/RL Component Training Script
=================================

Components:
1. VAE Zero-Day Detector - Normal trafik profili öğrenme
2. RL Threshold Agent - Adaptive alert threshold

Kullanım:
    python scripts/train_ai_components.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Proje yolu
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__}")

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017_full"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"


# ================== VAE ZERO-DAY DETECTOR ==================


class VAEZeroDay(keras.Model):
    """Variational Autoencoder for Zero-Day Detection"""

    def __init__(self, input_dim, latent_dim=32, beta=4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = keras.Sequential(
            [
                keras.layers.Dense(128, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(latent_dim * 2),  # mean and log_var
            ]
        )

        # Decoder
        self.decoder = keras.Sequential(
            [
                keras.layers.Dense(64, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(input_dim, activation="sigmoid"),
            ]
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, log_var = tf.split(h, 2, axis=-1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_recon = self.decode(z)
        return x_recon, mean, log_var

    def compute_loss(self, x, x_recon, mean, log_var):
        # Reconstruction loss
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(x, x_recon), axis=-1)
        )

        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)
        )

        return recon_loss + self.beta * kl_loss


def load_normal_traffic():
    """Sadece normal trafik yükle (VAE için)"""
    print("\n" + "=" * 60)
    print("📊 Loading Normal Traffic for VAE Training")
    print("=" * 60)

    train_file = DATA_DIR / "Train_data.csv"
    df = pd.read_csv(train_file, low_memory=False)

    # Sadece normal trafik
    df_normal = df[df["class"] == "normal"].copy()
    print(f"   Normal samples: {len(df_normal):,}")

    # Features
    feature_cols = [
        c for c in df.columns if c not in ["class", "protocol_type", "service", "flag"]
    ]

    for col in feature_cols:
        df_normal[col] = pd.to_numeric(df_normal[col], errors="coerce")
    df_normal[feature_cols] = df_normal[feature_cols].fillna(0)

    X = df_normal[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Scale
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    print(f"   Train: {len(X_train):,}")
    print(f"   Test: {len(X_test):,}")
    print(f"   Features: {X.shape[1]}")

    return X_train, X_test, scaler


def train_vae():
    """VAE eğit"""
    print("\n" + "=" * 60)
    print("🧠 Training VAE Zero-Day Detector")
    print("=" * 60)

    X_train, X_test, scaler = load_normal_traffic()

    # Model
    input_dim = X_train.shape[1]
    vae = VAEZeroDay(input_dim=input_dim, latent_dim=32, beta=4.0)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # Training
    EPOCHS = 50
    BATCH_SIZE = 128

    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")

    best_loss = float("inf")
    patience_counter = 0
    PATIENCE = 10

    print("\n🚀 Starting training...")

    for epoch in range(EPOCHS):
        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]

        epoch_loss = 0
        n_batches = len(X_train) // BATCH_SIZE

        for i in range(n_batches):
            batch = X_shuffled[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

            with tf.GradientTape() as tape:
                x_recon, mean, log_var = vae(batch)
                loss = vae.compute_loss(batch, x_recon, mean, log_var)

            grads = tape.gradient(loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(grads, vae.trainable_variables))
            epoch_loss += loss.numpy()

        epoch_loss /= n_batches

        # Validation
        x_recon_test, mean_test, log_var_test = vae(X_test)
        val_loss = vae.compute_loss(
            X_test, x_recon_test, mean_test, log_var_test
        ).numpy()

        if (epoch + 1) % 10 == 0:
            print(
                f"   Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}"
            )

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    # Calculate reconstruction error threshold
    x_recon_test, _, _ = vae(X_test)
    recon_errors = np.mean(np.square(X_test - x_recon_test.numpy()), axis=1)
    threshold = np.percentile(recon_errors, 95)

    print(f"\n   ✅ Training complete!")
    print(f"   📊 Reconstruction threshold (95th percentile): {threshold:.6f}")
    print(f"   📊 Mean recon error: {np.mean(recon_errors):.6f}")

    return vae, threshold, scaler


# ================== RL THRESHOLD AGENT ==================


class RLThresholdAgent:
    """Simple DQN Agent for Alert Threshold Optimization"""

    def __init__(self, state_dim=10, n_actions=5, lr=0.001):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.memory = []
        self.batch_size = 32
        self.max_memory = 10000

        # Q-Network
        self.model = keras.Sequential(
            [
                keras.layers.Dense(64, activation="relu", input_shape=(state_dim,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(n_actions),
            ]
        )
        self.model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")

        # Target Network
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)

        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]

            target = reward
            if not done:
                target += self.gamma * np.max(
                    self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                )

            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target

            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())


def simulate_ids_environment(n_samples=1000):
    """IDS ortamı simülasyonu"""
    # State: [mean_confidence, std_confidence, attack_rate, fp_rate, fn_rate, ...]
    states = np.random.rand(n_samples, 10).astype(np.float32)

    # True labels (0: normal, 1: attack)
    labels = (np.random.rand(n_samples) > 0.7).astype(int)

    # Confidence scores
    confidences = np.random.rand(n_samples)

    return states, labels, confidences


def train_rl_agent():
    """RL Agent'ı eğit"""
    print("\n" + "=" * 60)
    print("🤖 Training RL Threshold Agent")
    print("=" * 60)

    agent = RLThresholdAgent()

    EPISODES = 100
    STEPS_PER_EPISODE = 100

    print(f"   Episodes: {EPISODES}")
    print(f"   Steps per episode: {STEPS_PER_EPISODE}")

    rewards_history = []

    print("\n🚀 Starting training...")

    for episode in range(EPISODES):
        states, labels, confidences = simulate_ids_environment(STEPS_PER_EPISODE)

        episode_reward = 0
        threshold = 0.5  # Initial threshold

        for step in range(STEPS_PER_EPISODE):
            state = states[step]

            # Action: adjust threshold (-0.1, -0.05, 0, +0.05, +0.1)
            action = agent.act(state)
            threshold_delta = (action - 2) * 0.05
            threshold = np.clip(threshold + threshold_delta, 0.1, 0.9)

            # Simulate prediction
            prediction = 1 if confidences[step] > threshold else 0
            true_label = labels[step]

            # Reward
            if prediction == true_label:
                if prediction == 1:  # True positive
                    reward = 1.0
                else:  # True negative
                    reward = 0.5
            else:
                if prediction == 1:  # False positive
                    reward = -1.0
                else:  # False negative (worst!)
                    reward = -5.0

            next_state = states[(step + 1) % STEPS_PER_EPISODE]
            done = step == STEPS_PER_EPISODE - 1

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            episode_reward += reward

        if (episode + 1) % 10 == 0:
            agent.update_target()

        rewards_history.append(episode_reward)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            print(
                f"   Episode {episode+1}/{EPISODES} - Avg Reward: {avg_reward:.2f} - Epsilon: {agent.epsilon:.4f}"
            )

    print(f"\n   ✅ Training complete!")
    print(f"   📊 Final avg reward: {np.mean(rewards_history[-10:]):.2f}")

    return agent


# ================== SAVE MODELS ==================


def save_models(vae, vae_threshold, rl_agent):
    """Modelleri kaydet"""
    MODELS_DIR.mkdir(exist_ok=True)

    # VAE
    vae_id = f"vae_zero_day_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vae_path = MODELS_DIR / f"{vae_id}.keras"
    vae.save_weights(str(vae_path.with_suffix(".weights.h5")))

    # RL Agent
    rl_id = f"rl_threshold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rl_path = MODELS_DIR / f"{rl_id}.keras"
    rl_agent.model.save(rl_path)

    print(f"\n💾 VAE saved: {vae_path.with_suffix('.weights.h5')}")
    print(f"💾 RL Agent saved: {rl_path}")

    # Registry
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    registry["models"].append(
        {
            "id": vae_id,
            "name": "VAE_Zero_Day_Detector",
            "model_type": "vae",
            "status": "trained",
            "metrics": {"threshold": float(vae_threshold)},
            "created_at": datetime.now().isoformat(),
        }
    )

    registry["models"].append(
        {
            "id": rl_id,
            "name": "RL_Threshold_Agent",
            "model_type": "rl_agent",
            "status": "trained",
            "created_at": datetime.now().isoformat(),
        }
    )

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print("📋 Registry updated")


def main():
    print("\n" + "=" * 70)
    print("🎓 VAE/RL Component Training")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Train VAE
    vae, vae_threshold, scaler = train_vae()

    # Train RL Agent
    rl_agent = train_rl_agent()

    # Save
    save_models(vae, vae_threshold, rl_agent)

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
