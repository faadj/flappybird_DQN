import argparse
import os
import shutil
import time
from datetime import datetime
from random import random, randint, sample
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import imageio
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.01)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)

    # --- CHANGE 1: INCREASE TOTAL ITERATIONS TO 4 MILLION ---
    parser.add_argument("--num_iters", type=int, default=4000000)
    # --------------------------------------------------------

    parser.add_argument("--replay_memory_size", type=int, default=50000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--plot_path", type=str, default="training_plots")

    parser.add_argument("--vis", action="store_true", help="Turn on AI Vision visualization")
    parser.add_argument("--gif_interval", type=int, default=50000)
    parser.add_argument("--gif_path", type=str, default="training_gifs")

    args = parser.parse_args()
    return args


def update_plots(scores, losses, success_rates, filepath):
    if not scores: return
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory): os.makedirs(directory)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    episodes = range(1, len(scores) + 1)

    # 1. Score
    ax1.scatter(episodes, scores, s=10, c='black', alpha=0.6, label='Score')
    if len(scores) > 10:
        window = max(5, len(scores) // 20)
        trend = np.convolve(scores, np.ones(window) / window, mode='valid')
        ax1.plot(range(window, len(scores) + 1), trend, color='blue', linewidth=2, label='Trend')
    ax1.set_ylabel('Score')
    ax1.grid(True)

    # 2. Loss
    ax2.semilogy(episodes, losses, color='orange', linewidth=1, label='Avg Loss')
    ax2.set_ylabel('Avg Loss')
    ax2.grid(True, which="both", ls="-")

    # 3. Success Rate
    ax3.plot(episodes, success_rates, color='green', linewidth=2, label='Success Rate (>5 pipes)')
    ax3.set_ylabel('Success Rate %')
    ax3.set_xlabel('Episode')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)
    print(f"File saved to location: {filepath}")


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork()
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.exists(opt.saved_path):
        os.makedirs(opt.saved_path)

    writer = SummaryWriter(opt.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    game_state = FlappyBird()

    image, reward, terminal, state = game_state.next_frame(0)
    state = torch.from_numpy(state).float().unsqueeze(0)

    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = []
    iter = 0

    game_scores = []
    game_losses = []
    success_rates = []
    recent_successes = []

    current_game_score = 0
    current_game_loss_sum = 0
    current_game_steps = 0

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(opt.plot_path, f"training_session_{session_id}.png")

    gif_frames = []
    recording_gif = False
    if not os.path.exists(opt.gif_path): os.makedirs(opt.gif_path)

    last_log_time = time.time()
    last_plot_time = time.time()

    print(f"Training started (Raycast Lidar Mode). Vis: {'ON' if opt.vis else 'OFF'}")

    while iter < opt.num_iters:
        # --- SEND PROGRESS TO GAME (For Display) ---
        game_state.training_progress = iter / opt.num_iters
        # -------------------------------------------

        # --- CHANGE 2: SLOWER CURRICULUM ---
        # 0 - 200k: Easy Mode (Gap 200)
        # 200k - 3M: Very Slow Transition (Gap 200 -> 100)
        # 3M+: Hard Mode (Gap 100)

        if iter < 200000:
            game_state.pipe_gap_size = 200
            game_state.vertical_variance = 0.0
        elif iter < 3000000:  # Extended from 1M to 3M
            # Duration is now 2,800,000 steps (3M - 200k)
            progress = (iter - 200000) / 2800000
            game_state.pipe_gap_size = int(200 - (100 * progress))
            game_state.vertical_variance = progress
        else:
            game_state.pipe_gap_size = 100
            game_state.vertical_variance = 1.0
        # -------------------------------------------

        prediction = model(state)[0]
        epsilon = opt.final_epsilon + (
                (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        u = random()
        random_action = u <= epsilon
        if random_action:
            action = randint(0, 1)
        else:
            action = torch.argmax(prediction).item()

        image, reward, terminal, next_state_np = game_state.next_frame(action)

        next_state = torch.from_numpy(next_state_np).float().unsqueeze(0)
        if torch.cuda.is_available():
            next_state = next_state.cuda()

        if iter > 0 and iter % opt.gif_interval == 0:
            recording_gif = True
            gif_frames = []
        if recording_gif:
            frame_rgb = np.transpose(image, (1, 0, 2))
            gif_frames.append(frame_rgb)
            if len(gif_frames) >= 150:
                gif_filename = os.path.join(opt.gif_path, f"iter_{iter}.gif")
                try:
                    imageio.mimsave(gif_filename, gif_frames, fps=30)
                except Exception as e:
                    print(f"GIF Error: {e}")
                recording_gif = False
                gif_frames = []

        if opt.vis:
            view = cv2.cvtColor(np.transpose(image, (1, 0, 2)), cv2.COLOR_RGB2BGR)
            bx = int(game_state.bird_x + game_state.bird_width / 2)
            by = int(game_state.bird_y + game_state.bird_height / 2)

            # Draw Lidar Fan
            lidar_scope = 120
            start_angle = -lidar_scope / 2
            step = lidar_scope / 15

            for i in range(16):
                dist_norm = next_state_np[i]
                dist_px = dist_norm * 300

                deg = start_angle + (step * i)
                rad = math.radians(deg)

                ex = int(bx + math.cos(rad) * dist_px)
                ey = int(by + math.sin(rad) * dist_px)

                color = (0, 255, 0) if dist_norm > 0.5 else (0, 0, 255)
                cv2.line(view, (bx, by), (ex, ey), color, 1)

            cv2.imshow("Lidar Vision", view)
            cv2.waitKey(1)

        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]

        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(state_batch)
        next_state_batch = torch.cat(next_state_batch)

        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iter += 1

        if reward >= 1: current_game_score += 1
        current_game_loss_sum += loss.item()
        current_game_steps += 1

        if terminal:
            game_scores.append(current_game_score)
            avg_loss = current_game_loss_sum / current_game_steps if current_game_steps > 0 else 0
            game_losses.append(avg_loss)

            is_success = 1 if current_game_score >= 5 else 0
            recent_successes.append(is_success)
            if len(recent_successes) > 100: recent_successes.pop(0)
            current_success_rate = sum(recent_successes) / len(recent_successes) * 100
            success_rates.append(current_success_rate)

            current_game_score = 0
            current_game_loss_sum = 0
            current_game_steps = 0

        current_time = time.time()
        if current_time - last_log_time >= 30:
            print(
                f"Iter: {iter}/{opt.num_iters} | Gap: {game_state.pipe_gap_size} | SR: {success_rates[-1] if success_rates else 0:.1f}% | Loss: {loss.item():.6f}")
            last_log_time = current_time

        if current_time - last_plot_time >= 60:
            update_plots(game_scores, game_losses, success_rates, plot_filename)
            last_plot_time = current_time

        writer.add_scalar('Train/Loss', loss, iter)
        writer.add_scalar('Train/Reward', reward, iter)

        if (iter + 1) % 1000000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter + 1))

    torch.save(model, "{}/flappy_bird".format(opt.saved_path))


if __name__ == "__main__":
    plt.switch_backend('agg')
    opt = get_args()
    train(opt)