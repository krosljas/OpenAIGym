import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

#print("Tensorflow:", tf.__version__)

class DQNAgent:
    """
    DQN Agent: the agent that explores the game and
    should eventually learn how to play the game.
    """
    def __init__(self):
        """
        Prevents Q network from chasing a moving target by
        updating the target network only after a round of
        training is complete
        """
        self.q_net = self._build_dqn_model()
        self.target_q_net = self._build_dqn_model()

    @staticmethod
    def _build_dqn_model():
        """
        Builds a deep neural net which predicts the Q values for all possible
        actions given a state. The input should have the shape of the state
        (which is 4 in CartPole), and the output should have the same shape as
        the action space (which is 2 in CartPole) since we want 1 Q value per
        possible action.
        """

        q_net = Sequential()
        q_net.add(Dense(64, input_dim=4, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(2, activation='linear', kernel_initializer='he_uniform'))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mse')
        return q_net

    def random_policy(self, state):
        """
        Outputs a random action
        """
        return np.random.randint(0, 2)

    def collect_policy(self, state):
        """
        Similar to policy but with some randomness to encourage exploration
        """
        if np.random.random() < 0.05:
            return self.random_policy(state)
        return self.policy(state)

    def policy(self, state):
        """
        Takes a state from the game environment and returns
        an action that should be taken given the current game
        environment.
        """
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def update_target_network(self):
        """
        Updates the current target_q_net with the q_net which brings
        all the training in the q_net to the target_q_net
        """
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, batch):
        """
        Takes a batch of gameplay experiences from replay
        buffer and train the underlying model with the batch
        """
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = batch
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch).numpy()
        max_next_q = np.amax(next_q, axis=1)
        for i in range(state_batch.shape[0]):
            target_q_val = reward_batch[i]
            if not done_batch[i]:
                target_q_val += 0.95 * max_next_q[i]
            target_q[i][action_batch[i]] = target_q_val
        training_history = self.q_net.fit(x=state_batch, y=target_q, verbose=0)
        loss = training_history.history['loss']
        return loss
