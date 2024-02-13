import numpy as np
import tensorflow as tf
import random

SEED = 0
MINIBATCH_SIZE = 64
TAU = 1e-3
E_DECAY = 0.995
E_MIN = 0.01

random.seed(SEED)

def update_eps(epsilon):
    return max(E_MIN, E_DECAY*epsilon)

def epsilon_greedy_policy_action(q_values, eps = E_MIN):
    if random.random() > eps:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.argmax(4))
    
def update_target_network(q_network, target_q_network):
    for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
        target_weights.assign( TAU * q_net_weights + (1.0 - TAU) * target_weights)

def get_training_sample(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]), dtype=tf.float32
    )

    return (states, actions, rewards, next_states, done_vals)

def should_i_update(t, num_steps_upd, memory_buffer):
    return ((t+1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE)