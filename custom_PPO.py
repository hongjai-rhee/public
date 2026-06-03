import numpy as np
import tensorflow as tf

def get_log_prob(mu, std, action):
    """연속 행동에 대한 가우시안 로그 확률 계산"""
    var = tf.square(std)
    log_scale = tf.math.log(std + 1e-6) + 0.5 * tf.math.log(2.0 * np.pi)
    log_prob = -tf.square(action - mu) / (2.0 * var + 1e-6) - log_scale
    return tf.reduce_sum(log_prob, axis=-1)

def compute_gae_and_targets(rewards, values, next_values, dones, gamma=0.99, lmbda=0.95):
    """GAE(Advantage) 및 Critic의 target_value 일괄 계산"""
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
    """통합형 액터와 크리틱 모델을 전달받아 미니배치 기반 고속 업데이트를 수행 (엔트로피 제거 버전)"""
    def __init__(self, actor_model, critic_model, actor_lr=3e-4, critic_lr=1e-3, epsilon=0.2):
        self.actor_model = actor_model
        self.critic = critic_model

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.epsilon = epsilon

    @tf.function(reduce_retracing=True)
    def train_step(self, states_t, actions_t, old_log_probs_t, advantages_t, target_values_t):
        """Persistent GradientTape를 활용한 액터-크리틱 통합 고속 업데이트 단계"""
        with tf.GradientTape(persistent=True) as tape:
            # 1. Actor Update 로직 (듀얼 헤드로부터 mu와 log_std를 배치 단위로 획득)
            new_mu, log_std = self.actor_model(states_t, training=True)
            
            # 지수 폭발 방지 및 미세 제어를 위한 최종 클리핑 가이드 안정선
            log_std = tf.clip_by_value(log_std, -5.0, 2.0)
            new_std = tf.exp(log_std)
            
            new_log_probs = get_log_prob(new_mu, new_std, actions_t)

            # 중요도 샘플링(Importance Sampling Ratio) 및 클리핑 로스 연산
            ratio = tf.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages_t
            
            # 🔥 [수정] 엔트로피 보너스 항을 완전히 제거하여 순수한 PPO 클리핑 로스만 남김
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # 2. Critic Update 로직
            predicted_values = tf.squeeze(self.critic(states_t, training=True), axis=-1)
            critic_loss = tf.reduce_mean(tf.square(target_values_t - predicted_values))

        # 가중치 갱신
        actor_vars = self.actor_model.trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_vars)
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

        critic_vars = self.critic.trainable_variables
        critic_grads = tape.gradient(critic_loss, critic_vars)
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

        del tape
        return actor_loss, critic_loss
     
    def update(self, states, actions, old_log_probs, advantages, target_values, epochs=4, batch_size=64):
        """기존 넘파이 데이터를 텐서로 포맷팅하고, 미니배치로 분할하여 고속 반복 업데이트 수행"""        
        states = np.array(states)
        actions = np.array(actions)
        old_log_probs = np.array(old_log_probs)
        advantages = np.array(advantages)
        target_values = np.array(target_values)

        # 정규화(Advantage) 처리를 통해 PPO 확률 그라디언트 안정화
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        data_size = len(states)
        
        for _ in range(epochs):
            indices = np.arange(data_size)
            np.random.shuffle(indices)
            
            for start in range(0, data_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                states_t = tf.convert_to_tensor(states[batch_idx], dtype=tf.float32)
                actions_t = tf.convert_to_tensor(actions[batch_idx], dtype=tf.float32)
                old_log_probs_t = tf.convert_to_tensor(old_log_probs[batch_idx], dtype=tf.float32)
                advantages_t = tf.convert_to_tensor(advantages[batch_idx], dtype=tf.float32)
                target_values_t = tf.convert_to_tensor(target_values[batch_idx], dtype=tf.float32)
                
                self.train_step(states_t, actions_t, old_log_probs_t, advantages_t, target_values_t)