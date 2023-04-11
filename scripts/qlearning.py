"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
import random
import torch.optim as optim

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from classifier_sample import (
    gen,
    model_fn,
    create_cond_fn
)
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QLearningEnvironment:
    def __init__(self, cnn, guided_diffusion_model, input_image):
        self.cnn = cnn
        self.guided_diffusion_model = guided_diffusion_model
        self.input_image = input_image
        self.generated_image = None

    def reset(self):
        # Generate an initial image using the guided diffusion model
        self.generated_image = self.guided_diffusion_model.generate(self.input_image)
        return (self.input_image, self.generated_image)

    def step(self, action):
        if action == 0:  # Generate a new image
            new_generated_image = self.guided_diffusion_model.generate(self.input_image)
        elif action == 1:  # Keep the current generated image
            new_generated_image = self.generated_image
        else:
            raise ValueError("Invalid action")

        # Calculate the reward as the improvement in the CNN training accuracy
        reward = self.calculate_reward(self.generated_image, new_generated_image)
        self.generated_image = new_generated_image
        return (self.input_image, self.generated_image), reward

def q_learning_train_step(q_network, optimizer, memory, batch_size, discount_factor):
    if len(memory) < batch_size:
        return

    # Sample a batch of transitions from the memory
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states = zip(*batch)
    states = th.stack(states)
    actions = th.tensor(actions)
    rewards = th.tensor(rewards)
    next_states = th.stack(next_states)

    # Compute the Q-values for the current states and actions
    q_values = q_network(states).gather(1, actions.unsqueeze(1))

    # Compute the target Q-values for the next states
    next_q_values = q_network(next_states).max(1)[0]
    target_q_values = rewards + discount_factor * next_q_values

    # Update the Q-network weights
    loss = F.mse_loss(q_values, target_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def create_argparser():
    defaults = dict(
        similarity_scale=100,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


memory = []
learning_rate = 1e-4
discount_factor = 0.99
batch_size = 64
update_interval = 10

q_network = QNetwork().to(dist_util.dev())
q_learning_env = QLearningEnvironment(q_network, model_fn, )
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

input_image_path = "imgs/j.png"
input_image = Image.open(input_image_path).convert("RGB")
input_image = input_image.resize((args.image_size, args.image_size))
input_image = th.from_numpy(np.array(input_image)).permute(2, 0, 1).float()
input_image = (input_image / 127.5) - 1.0  # Normalize to the range [-1, 1]


# Sampling with Q-learning feedback
step_count = 0
while len(all_images) * args.batch_size < args.num_samples:
    state = q_learning_env.reset()
    done = False

    while not done:
        # Select an action using the Q-network
        q_values = q_network(th.tensor(state).unsqueeze(0).to(dist_util.dev()))
        action = th.argmax(q_values).item()
        next_state, reward = q_learning_env.step(action)
        memory.append((th.tensor(state), action, reward, th.tensor(next_state)))
        if step_count % update_interval == 0:
            q_learning_train_step(q_network, optimizer, memory, batch_size, discount_factor)

        state = next_state
        step_count += 1

        # Check if the generated image meets the "good" criteria
        if reward == 1:
            done = True
            # Save the generated image and its label
            all_images.append(state[1].cpu().numpy())
            all_labels.append(input_class_label.cpu().numpy())
            logger.log(f"created {len(all_images) * args.batch_size} samples")


arr = np.concatenate(all_images, axis=0)
arr = arr[: args.num_samples]
label_arr = np.concatenate(all_labels, axis=0)
label_arr = label_arr[: args.num_samples]
if dist.get_rank() == 0:
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join("/path/to/output", f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, label_arr)


if __name__ == "__main__":
    args = create_argparser().parse_args()