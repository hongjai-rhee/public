import numpy as np
import tensorflow as tf

def get_log_prob(mu, std, action):
    """안정적인 가우시안 로그 확률 계산"""
    var = tf.square(std)
    log_scale = tf.math.log(std + 1e-8) + 0.5 * tf.math.log(2.0 * np.pi)
    log_prob = -tf.square(action - mu) / (2.0 * var + 1e-8) - log_scale
    return tf.reduce_sum(log_prob, axis=-1)

def compute_gae_and_targets(rewards, values, next_values, dones, gamma=0.99, lmbda=0.95):
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        td_target = rewards[t] + gamma * next_values[t] * mask
        delta = td_target - values[t]
        gae = delta + gamma * lmbda * mask * gae
        advantages[t] = gae
    return advantages, advantages + values

class PPOUpdater:
    def __init__(self, actor_model, critic_model, actor_lr=3e-4, critic_lr=1e-3, epsilon=0.1):
        self.actor_model = actor_model
        self.critic = critic_model
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr, clipnorm=1.0) # 그라디언트 폭발 방지
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, clipnorm=1.0)
        self.epsilon = epsilon

    @tf.function(reduce_retracing=True)
    def train_step(self, states_t, actions_t, old_log_probs_t, advantages_t, target_values_t):
        with tf.GradientTape(persistent=True) as tape:
            new_mu, log_std = self.actor_model(states_t, training=True)
            
            # 그라디언트가 살아있는 상태에서 안전하게 제한 (-2.0은 std 0.13, -0.5는 std 0.6)
            log_std = tf.clip_by_value(log_std, -3.5, -0.5)
            new_std = tf.exp(log_std)
            
            new_log_probs = get_log_prob(new_mu, new_std, actions_t)

            ratio = tf.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages_t
            
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            predicted_values = tf.squeeze(self.critic(states_t, training=True), axis=-1)
            critic_loss = tf.reduce_mean(tf.square(target_values_t - predicted_values))

        actor_vars = self.actor_model.trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_vars)
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

        critic_vars = self.critic.trainable_variables
        critic_grads = tape.gradient(critic_loss, critic_vars)
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

        del tape
        return actor_loss, critic_loss

    def update(self, states, actions, old_log_probs, advantages, target_values, epochs=4, batch_size=64):
        states = np.array(states)
        actions = np.array(actions)
        old_log_probs = np.array(old_log_probs)
        advantages = np.array(advantages)
        target_values = np.array(target_values)

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        data_size = len(states)
        
        for _ in range(epochs):
            indices = np.arange(data_size)
            np.random.shuffle(indices)
            
            for start in range(0, data_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                self.train_step(
                    tf.convert_to_tensor(states[batch_idx], dtype=tf.float32),
                    tf.convert_to_tensor(actions[batch_idx], dtype=tf.float32),
                    tf.convert_to_tensor(old_log_probs[batch_idx], dtype=tf.float32),
                    tf.convert_to_tensor(advantages[batch_idx], dtype=tf.float32),
                    tf.convert_to_tensor(target_values[batch_idx], dtype=tf.float32)
                )