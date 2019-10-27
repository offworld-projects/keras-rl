from __future__ import division
from collections import deque
import os
import warnings
import math

import numpy as np
import keras.backend as K
import keras.optimizers as optimizers

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001, **kwargs):
        if hasattr(actor.output, '__len__') and len(actor.output) > 1:
            raise ValueError('Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError('Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
        if critic_action_input not in critic.input:
            raise ValueError('Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError('Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.format(critic))

        super(DDPGAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        state_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                state_inputs.append(i)
        combined_inputs[self.critic_action_input_idx] = self.actor(state_inputs)

        combined_output = self.critic(combined_inputs)

        updates = actor_optimizer.get_updates(
            params=self.actor.trainable_weights, loss=-K.mean(combined_output))
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(state_inputs + [K.learning_phase()],
                                             [self.actor(state_inputs)], updates=updates)
        else:
            if self.uses_learning_phase:
                state_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(state_inputs, [self.actor(state_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:]
                else:
                    inputs = [state0_batch]
                if self.uses_learning_phase:
                    inputs += [self.training]
                action_values = self.actor_train_fn(inputs)[0]
                assert action_values.shape == (self.batch_size, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics

class HALGANDDPGAgent(DDPGAgent):

    def select_action(self, state):
        img_batch, config_batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(img_batch).flatten()

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            action += noise
        # now clip the action
        action = np.clip(action, a_max=self.action_box[1], a_min=self.action_box[0])
        # sometime action is just epsilon greedy
        random_action = np.random.uniform(low=self.action_box[0], high=self.action_box[1])
        action += np.random.binomial(1, self.eps(self.step), action.shape[0])*(random_action - action)
        return action

    def configure_gan(self, generator, latent_size, filepath):
        self.generator = generator
        self.generator.load_weights(filepath)
        self.gan_latent_size = latent_size

    def convert_config(self, config_current, config_final):
        if self.ENV_NAME == 'MiniWorld-SimToReal2Cont-v0':
            x1, y1, yaw1, grip1 = config_final
            x2, y2, yaw2, grip2 = config_current
        if self.ENV_NAME == 'MiniWorld-SimToReal1Cont-v0':
            x1, y1, yaw1 = config_final
            x2, y2, yaw2 = config_current
        dist = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
        if abs(x2-x1) < 1e-3:
            if abs(y2-y1) < 1e-3:
                dist=0.
                angle = yaw2-yaw1
                if angle < 0:
                    angle += 2*math.pi
                if angle > math.pi:
                    angle -= 2*math.pi
                if self.ENV_NAME == 'MiniWorld-SimToReal2Cont-v0':
                    return dist, angle, 1-(grip1-grip2)  # it's the same x,y location
                else:
                    return dist, angle  # it's the same x,y location
            if y2 > (y1 + .1):
                theta = 3*math.pi/2
            else:
                theta = math.pi/2
        else:
            theta = math.atan((y1-y2)/(x1-x2))
        # first convert theta to [0,2pi]
        if x1 < x2:
            theta += math.pi
        if theta < 0:
            theta += 2*math.pi
        angle = theta - yaw2 # relative angle of viewing the goal
        # center it [-pi, pi]
        if angle < 0:
            angle += 2*math.pi
        if angle > math.pi:
            angle -= 2*math.pi
        if self.ENV_NAME == 'MiniWorld-SimToReal2Cont-v0':
            return dist, angle, 1-(grip1-grip2)
        else:
            return dist, angle #it's the same x,y location

    def generate_hallucinations(self, chunk):
        '''
        arguments:
            chunk: list of [states, actions] of failed transitions
            that pass the acceptance criteria
        '''
        fail0, config0 = zip(*[chunk[i][0][0] for i in range(len(chunk))])
        fail0 = np.array(fail0)
        config0 = np.array(config0)
        fail1, config1 = zip(*[chunk[i][0][1] for i in range(len(chunk))])
        fail1 = np.array(fail1)
        config1 = np.array(config1)
        fail_last, config_last = zip(*[chunk[i][0][-1] for i in range(len(chunk))])
        fail_last = np.array(fail_last)
        config_last = np.array(config_last)
        if self.mode == 'halgan':
            # GANs are trained with states in range [-1,1], but states here
            # are [0,1], so we convert back and forth
            fail0 = (fail0*2)-1
            fail1 = (fail1*2)-1
            # get relative config to last state in chunk
            config0 = np.array([self.convert_config(config0[i,:], config_last[i,:]) for i in range(len(chunk))])
            config1 = np.array([self.convert_config(config1[i,:], config_last[i,:]) for i in range(len(chunk))])
            if self.ENV_NAME == 'MiniWorld-SimToReal2Cont-v0':
                # randomly decide what is being hallucinated
                config0[:,-1] = np.random.randint(0,2,size=config0.shape[0])
                config1[:,-1] = config0[:,-1]
            generated_images = self.generator.predict([
                np.random.normal(1., .1, (2*len(chunk), self.gan_latent_size)),
                np.concatenate((config0, config1), axis=0)])
            # add in the diffs to create states
            fake0 = fail0 + generated_images[0:len(chunk)]
            fake0 = np.tanh(fake0)
            fake1 = fail1 + generated_images[len(chunk):]
            fake1 = np.tanh(fake1)
            # now convert generated images back to [0,1]
            fake0 = (fake0+1)/2
            fake1 = (fake1+1)/2
            fake_done = np.zeros((len(chunk),))
            fake_done[self.fake_done_criteria(config1)] = 1.
            fake_reward = np.array([chunk[i][2] for i in range(len(chunk))])
            fake_reward[np.where(fake_done)[0]] = 1.
        elif self.mode == 'rig-':
            # the fakes are the same as the fails
            fake0=fail0.copy()
            fake1=fail1.copy()
            # the reward is decided by the encoder mean distance
            en0 = self.encoder.predict([fake0,])[0]
            en1 = self.encoder.predict([fake1,])[0]
            # compare distance to random images from near goal
            idxs = np.random.randint(0, len(self.near_goal), size=len(chunk))
            eng = self.encoder.predict(np.array([self.near_goal[i] for i in idxs]))[0]
            fake_reward = np.array([chunk[i][2] for i in range(len(chunk))])
            fake_reward += -0.1*np.linalg.norm(np.array([self.labels[i] for i in idxs]), axis=1)*np.linalg.norm((eng-en1),axis=1)
            fake_done = np.zeros((len(chunk),))
        elif self.mode == 'vae-her':
            # the fakes are the same as the fails
            fake0=fail0.copy()
            fake1=fail1.copy()
            fake_last = fail_last.copy()
            # the reward is decided by the encoder mean distance
            en0 = self.encoder.predict([fake0,])[0]
            en1 = self.encoder.predict([fake1,])[0]
            en_last = self.encoder.predict([fake_last,])[0]
            # compare distance to last image in trajectory
            fake_reward = np.array([chunk[i][2] for i in range(len(chunk))])
            fake_reward += -np.linalg.norm((en_last-en1),axis=1)
            fake_done = np.zeros((len(chunk),))
            config1 = np.array([self.convert_config(config1[i,:], config_last[i,:]) for i in range(len(chunk))])
            fake_done[self.fake_done_criteria(config1)] = 1.
        elif self.mode == 'her':
            # no modification to the images are done
            fake0=fail0.copy()
            fake1=fail1.copy()
            # everything else the same as vher
            # reward is assigned by relative configuration to final state
            config0 = np.array([self.convert_config(config0[i,:], config_last[i,:]) for i in range(len(chunk))])
            config1 = np.array([self.convert_config(config1[i,:], config_last[i,:]) for i in range(len(chunk))])
            fake_done = np.zeros((len(chunk),))
            fake_done[self.fake_done_criteria(config1)] = 1.
            fake_reward = np.array([chunk[i][2] for i in range(len(chunk))])
            fake_reward[np.where(fake_done)[0]] = 1.
        else:
            raise NotImplementedError
        fake_action = np.array([chunk[i][1] for i in range(len(chunk))])

        return fake0, fake1, fake_action, fake_reward, fake_done

    def fake_done_criteria(self, rel_config):
        if self.ENV_NAME == 'MiniWorld-SimToReal1Cont-v0':
            return np.where(np.logical_and(rel_config[:,0] < 0.01, abs(rel_config[:,1]) < 0.1))[0]
        elif self.ENV_NAME == 'MiniWorld-SimToReal2Cont-v0':
            return np.where(rel_config[:,-1]*np.logical_and(rel_config[:,0] < 0.01, abs(rel_config[:,1]) < 0.1))[0]

    def acceptance_criteria(self, states, rewards, terminals):
        '''
        check whether to accept sequence for hallucination.
        return: False if any of the states achieve the goal or terminate.
        False if starting state is too close to goal state.
        '''
        fail0, config0 = states[0]
        config0 = np.array(config0)
        fail1, config1 = states[1]
        config1 = np.array(config1)
        fail_last, config_last = states[-1]
        config_last = np.array(config_last)
        # get relative config to last state in chunk
        config0 = np.array(self.convert_config(config0, config_last))
        config1 = np.array(self.convert_config(config1, config_last))
        if config0[0]<0.01:
            return False
        elif any(terminals):
            return False
        elif any([r > 0 for r in rewards]):
            return False
        return True

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            # draw batch_size random numbers
            p = np.random.uniform(size=self.batch_size)
            num_hallucinated_samples = int(np.sum(p < self.percent_hallucination(self.step)/100.))
            experiences = self.memory.sample(self.batch_size - num_hallucinated_samples)
            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)
            state0_batch, config0_batch = self.process_state_batch(state0_batch)
            state1_batch, config1_batch = self.process_state_batch(state1_batch)

            real_rewards = np.sum(np.array(reward_batch))
            if num_hallucinated_samples > 0:
                # pick how many steps before goal transition do you wanna be?
                dist_to_goal = np.random.randint(0, self.max_dist_to_goal, size=num_hallucinated_samples)
                # now sample hallucinations
                chunks = self.memory.sample_failed_triplets(
                    num_hallucinated_samples,
                    dist_to_goal+1,
                    self.acceptance_criteria)
                fake0, fake1, fake_action, fake_reward, fake_done =\
                    self.generate_hallucinations(chunks)
                state0_batch = np.concatenate((state0_batch, fake0))
                state1_batch = np.concatenate((state1_batch, fake1))
                reward_batch = np.concatenate((reward_batch, fake_reward))
                action_batch = np.concatenate((action_batch, fake_action))
                terminal1_batch = np.concatenate((terminal1_batch, 1.-fake_done))

            hallucinated_rewards = np.sum(np.array(reward_batch)) - real_rewards
            # Prepare and validate parameters.
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics
                metrics += [real_rewards, hallucinated_rewards, self.eps(self.step),]

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:]
                else:
                    inputs = [state0_batch]
                if self.uses_learning_phase:
                    inputs += [self.training]
                action_values = self.actor_train_fn(inputs)[0]
                assert action_values.shape == (self.batch_size, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics
    @property
    def metrics_names(self):
        '''add the gan rewards related metrics'''
        return super(HALGANDDPGAgent, self).metrics_names + \
                ['real_sampled_rewards', 'hallucinated_sampled_rewards', 'mean eps',]
