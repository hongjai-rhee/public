"""
mini_PPO.py
===========
LunarLanderContinuous-v2 용 PPO (from scratch)
TensorFlow / Keras + tensorflow_probability

클래스 구조:
  ModelSetup        → Actor(μ, σ) + Critic(V) 네트워크
  TakeAction        → Gaussian 정책 샘플링 & log_prob
  RolloutCollector  → 환경 상호작용 & 궤적 버퍼 관리
  EvaluateAdvantage → GAE로 advantage & return 계산
  UpdateModel       → PPO Clipped Loss 업데이트
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. ModelSetup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ModelSetup:
    """
    Actor-Critic 네트워크를 정의합니다.

    [Actor]  상태 s → 행동의 평균(μ) 출력
             log_std는 상태 무관 전역 파라미터 (OpenAI 스타일)
             → Gaussian 정책: π(a|s) = N(μ, σ²)
    [Critic] 상태 s → 상태가치 V(s) 출력
    """

    def __init__(self, state_dim, action_dim, actor_lr=3e-4, critic_lr=1e-3):
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.actor  = self._build_actor()
        self.critic = self._build_critic()

        self.actor_optimizer  = keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(critic_lr)

    def _build_actor(self):
        inp = layers.Input(shape=(self.state_dim,))
        x   = layers.Dense(64, activation='tanh')(inp)
        x   = layers.Dense(64, activation='tanh')(x)
        mu  = layers.Dense(self.action_dim, activation='tanh')(x)

        log_std = tf.Variable(
            initial_value=-0.5 * np.ones(self.action_dim, dtype=np.float32),
            trainable=True,
            name='log_std'
        )
        model = keras.Model(inputs=inp, outputs=mu, name='Actor')
        model.log_std = log_std
        return model

    def _build_critic(self):
        inp = layers.Input(shape=(self.state_dim,))
        x   = layers.Dense(64, activation='tanh')(inp)
        x   = layers.Dense(64, activation='tanh')(x)
        val = layers.Dense(1)(x)
        return keras.Model(inputs=inp, outputs=val, name='Critic')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. TakeAction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TakeAction:
    """
    Gaussian 정책에서 행동 샘플링 & log_prob 계산.

    수식:
      π(a|s) = N(μ(s), σ²)
      log π(a|s) = Σ [ -0.5*(a-μ)²/σ² - log(σ) - 0.5*log(2π) ]
    """

    def __init__(self, model_setup: ModelSetup):
        self.actor = model_setup.actor

    def __call__(self, state: np.ndarray):
        """단일 상태 → (action, log_prob, mu) 반환 (env 스텝용)"""
        state_t  = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        mu       = self.actor(state_t)
        std      = tf.exp(self.actor.log_std)
        dist     = tfp.distributions.Normal(mu, std)
        action   = dist.sample()
        log_prob = tf.reduce_sum(dist.log_prob(action), axis=-1)
        action   = tf.clip_by_value(action, -1.0, 1.0)

        return (action.numpy()[0],
                log_prob.numpy()[0],
                mu.numpy()[0])

    def get_log_prob(self, states_t, actions_t):
        """배치용 — UpdateModel 내부에서 새 정책의 log_prob 계산"""
        mu   = self.actor(states_t)
        std  = tf.exp(self.actor.log_std)
        dist = tfp.distributions.Normal(mu, std)
        return tf.reduce_sum(dist.log_prob(actions_t), axis=-1)  # (batch,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. RolloutCollector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RolloutCollector:
    """
    환경과 상호작용하며 rollout_steps만큼 궤적을 수집합니다.
    에피소드 경계(done) 자동 처리, 로깅, numpy 변환까지 책임집니다.
    """

    def __init__(self, env, take_action: TakeAction, rollout_steps: int = 2048):
        self.env           = env
        self.take_action   = take_action
        self.rollout_steps = rollout_steps

        self.state, _  = env.reset()
        self.ep_return = 0.0
        self.episode   = 0
        self.step      = 0

    def collect(self) -> dict:
        """rollout_steps만큼 수집 후 trajectory dict 반환"""
        buf = {k: [] for k in ['states', 'actions', 'rewards', 'dones', 'log_probs']}

        for _ in range(self.rollout_steps):
            action, log_prob, _ = self.take_action(self.state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            buf['states'].append(self.state)
            buf['actions'].append(action)
            buf['rewards'].append(reward)
            buf['dones'].append(float(done))
            buf['log_probs'].append(log_prob)

            self.ep_return += reward
            self.step      += 1
            self.state      = next_state

            if done:
                self.episode += 1
                self._log(self.ep_return)
                self.ep_return = 0.0
                self.state, _ = self.env.reset()

        return {k: np.array(v, dtype=np.float32) for k, v in buf.items()}

    def _log(self, ep_return: float):
        print(f"  [Episode {self.episode:4d}] Reward: {ep_return:8.2f}  "
              f"Total_Steps: {self.step:7d}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. EvaluateAdvantage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EvaluateAdvantage:
    """
    Generalized Advantage Estimation (GAE).

    수식:
      δt = rt + γ·V(st+1) - V(st)        ← TD 잔차
      Ât = δt + γλ·Ât+1                  ← 역방향 재귀
      Gt = Ât + V(st)                     ← Critic 학습 타겟 (return)

    λ=1 → MC에 가까움(high variance)
    λ=0 → TD(low variance)
    """

    def __init__(self, model_setup: ModelSetup, gamma: float = 0.99, lam: float = 0.95):
        self.critic = model_setup.critic
        self.gamma  = gamma
        self.lam    = lam

    def __call__(self, trajectory: dict) -> dict:
        states  = trajectory['states']
        rewards = trajectory['rewards']
        dones   = trajectory['dones']

        values = self.critic(
            tf.convert_to_tensor(states, dtype=tf.float32)
        ).numpy().flatten()

        T          = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < T else 0.0
            mask     = 1.0 - dones[t]
            delta    = rewards[t] + self.gamma * next_val * mask - values[t]
            gae      = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae

        returns = advantages + values

        # Advantage 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        trajectory['advantages'] = advantages
        trajectory['returns']    = returns
        return trajectory


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. UpdateModel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class UpdateModel:
    """
    PPO Clipped Surrogate Objective로 Actor-Critic 업데이트.

    Actor Loss:
      L_CLIP = E[ min( r_t·Â_t, clip(r_t, 1-ε, 1+ε)·Â_t ) ]
      r_t = π_new(a|s) / π_old(a|s)

    Critic Loss:
      L_VF = MSE( V(s), G_t )

    Entropy Bonus:
      L_ENT = β·H[π]  (탐험 장려)

    최종: L = -L_CLIP + c1·L_VF - c2·L_ENT
    """

    def __init__(self, model_setup: ModelSetup, take_action: TakeAction,
                 clip_eps: float = 0.2, epochs: int = 10, batch_size: int = 64,
                 vf_coef: float = 0.5, ent_coef: float = 0.01):

        self.actor            = model_setup.actor
        self.critic           = model_setup.critic
        self.actor_optimizer  = model_setup.actor_optimizer
        self.critic_optimizer = model_setup.critic_optimizer
        self.take_action      = take_action

        self.clip_eps   = clip_eps
        self.epochs     = epochs
        self.batch_size = batch_size
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef

    def __call__(self, trajectory: dict) -> dict:
        states        = tf.constant(trajectory['states'],     dtype=tf.float32)
        actions       = tf.constant(trajectory['actions'],    dtype=tf.float32)
        advantages    = tf.constant(trajectory['advantages'], dtype=tf.float32)
        returns       = tf.constant(trajectory['returns'],    dtype=tf.float32)
        old_log_probs = tf.constant(trajectory['log_probs'],  dtype=tf.float32)

        T = states.shape[0]
        history = {'actor_loss': [], 'critic_loss': [], 'entropy': []}

        for _ in range(self.epochs):
            indices = np.random.permutation(T)

            for start in range(0, T, self.batch_size):
                # numpy 인덱스 → tf.constant 변환 후 tf.gather로 슬라이싱
                idx = tf.constant(indices[start: start + self.batch_size], dtype=tf.int32)

                # ✅ _update_step이 loss 값을 return → Python 레벨에서 append
                al, cl, ent = self._update_step(
                    tf.gather(states,        idx),
                    tf.gather(actions,       idx),
                    tf.gather(advantages,    idx),
                    tf.gather(returns,       idx),
                    tf.gather(old_log_probs, idx),
                )
                history['actor_loss'].append(al.numpy())
                history['critic_loss'].append(cl.numpy())
                history['entropy'].append(ent.numpy())

        return {k: float(np.mean(v)) for k, v in history.items()}

    @tf.function  # ✅ graph 컴파일로 속도 복원 — history 인자 제거로 호환
    def _update_step(self, s, a, adv, ret, old_lp):

        # ── Actor 업데이트 ──────────────────────────────────
        with tf.GradientTape() as tape_a:
            new_log_probs = self.take_action.get_log_prob(s, a)
            ratio      = tf.exp(new_log_probs - old_lp)
            surr1      = ratio * adv
            surr2      = tf.clip_by_value(ratio, 1 - self.clip_eps,
                                                  1 + self.clip_eps) * adv
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            std        = tf.exp(self.actor.log_std)
            entropy    = tf.reduce_mean(0.5 * tf.math.log(2 * np.pi * np.e * std ** 2))
            actor_loss = actor_loss - self.ent_coef * entropy

        actor_vars = self.actor.trainable_variables + [self.actor.log_std]
        grads_a    = tape_a.gradient(actor_loss, actor_vars)
        grads_a    = [tf.clip_by_norm(g, 0.5) for g in grads_a]
        self.actor_optimizer.apply_gradients(zip(grads_a, actor_vars))

        # ── Critic 업데이트 ─────────────────────────────────
        with tf.GradientTape() as tape_c:
            values      = tf.squeeze(self.critic(s), axis=-1)
            critic_loss = self.vf_coef * tf.reduce_mean((ret - values) ** 2)

        grads_c = tape_c.gradient(critic_loss, self.critic.trainable_variables)
        grads_c = [tf.clip_by_norm(g, 0.5) for g in grads_c]
        self.critic_optimizer.apply_gradients(
            zip(grads_c, self.critic.trainable_variables)
        )

        # ✅ history 대신 return으로 값 반환
        return actor_loss, critic_loss, entropy
