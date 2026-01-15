import numpy as np
import torch
from typing import Dict, List
import time
from collections import deque
import json

from WarehouseEnv import WarehouseEnv
from mappoimplementation import MAPPO


class MAPPOTrainer:
    """
    Training loop for MAPPO in warehouse environment.
    """

    def __init__(
            self,
            env: WarehouseEnv,
            mappo: MAPPO,
            n_steps: int = 2048,
            n_epochs: int = 4,
            batch_size: int = 64,
            save_interval: int = 100,
            log_interval: int = 10,
            save_dir: str = "./checkpoints"
    ):
        self.env = env
        self.mappo = mappo
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.save_dir = save_dir

        # Metrics
        self.episode_rewards = {agent: [] for agent in env.possible_agents}
        self.episode_lengths = []
        self.episode_deliveries = []
        self.episode_collisions = []

        self.reward_window = deque(maxlen=100)
        self.delivery_window = deque(maxlen=100)

    # --------------------------------------------------------
    # Rollout Collection
    # --------------------------------------------------------

    def collect_rollouts(self):
        step_count = 0
        episode_count = 0

        obs, _ = self.env.reset()
        episode_rewards = {agent: 0 for agent in self.env.agents}
        episode_steps = 0

        while step_count < self.n_steps:

            global_state = self.env.get_global_state()

            actions = {}
            agent_data = {}

            for agent in self.env.agents:
                action, log_prob, value = self.mappo.select_action(
                    obs[agent], global_state
                )

                actions[agent] = action
                agent_data[agent] = {
                    "obs": obs[agent],
                    "action": action,
                    "log_prob": log_prob,
                    "value": value
                }

            next_obs, rewards, terms, truncs, infos = self.env.step(actions)

            # ✅ FIXED: Render during training if in human mode
            if self.env.render_mode == "human":
                self.env.render()

            for agent in self.env.agents:
                done = terms[agent] or truncs[agent]

                self.mappo.buffer.add(
                    obs=agent_data[agent]["obs"],
                    action=agent_data[agent]["action"],
                    reward=rewards[agent],
                    done=done,
                    value=agent_data[agent]["value"],
                    log_prob=agent_data[agent]["log_prob"],
                    global_state=global_state
                )

                episode_rewards[agent] += rewards[agent]

            step_count += 1
            episode_steps += 1

            if any(terms.values()) or any(truncs.values()):
                avg_reward = np.mean(
                    [episode_rewards[a] for a in self.env.agents]
                )

                self.reward_window.append(avg_reward)
                self.episode_lengths.append(episode_steps)

                for a in self.env.agents:
                    self.episode_rewards[a].append(episode_rewards[a])

                metrics = self.env.world.get_metrics()
                self.episode_deliveries.append(metrics["total_deliveries"])
                self.delivery_window.append(metrics["total_deliveries"])
                self.episode_collisions.append(metrics["collision_count"])

                obs, _ = self.env.reset()
                episode_rewards = {agent: 0 for agent in self.env.agents}
                episode_steps = 0
                episode_count += 1
            else:
                obs = next_obs

        return episode_count

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    def train(self, total_timesteps: int):

        print("=" * 70)
        print("Starting MAPPO Training")
        print("=" * 70)

        timesteps = 0
        update_count = 0
        start_time = time.time()

        while timesteps < total_timesteps:

            print(f"\n[Update {update_count + 1}] Collecting rollouts...")
            episode_count = self.collect_rollouts()
            timesteps += self.n_steps

            print(f"[Update {update_count + 1}] Updating networks...")
            train_metrics = self.mappo.update(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size
            )

            update_count += 1

            if update_count % self.log_interval == 0:

                elapsed = time.time() - start_time
                fps = timesteps / elapsed

                print("\n" + "=" * 70)
                print(f"Update {update_count}")
                print(f"Timesteps: {timesteps:,}")
                print(f"FPS: {fps:.1f}")

                if self.reward_window:
                    print(f"Mean Reward(100): {np.mean(self.reward_window):.2f}")

                if self.delivery_window:
                    print(f"Mean Deliveries(100): {np.mean(self.delivery_window):.2f}")

                print("Losses:")
                print(f"Policy: {train_metrics['policy_loss']:.4f}")
                print(f"Value: {train_metrics['value_loss']:.4f}")
                print(f"Entropy: {train_metrics['entropy']:.4f}")
                print("=" * 70)

            if update_count % self.save_interval == 0:
                self.save_checkpoint(update_count, timesteps)

        print("\nTraining completed!")
        self.save_checkpoint(update_count, timesteps, final=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    def save_checkpoint(self, update_count, timesteps, final=False):

        import os
        os.makedirs(self.save_dir, exist_ok=True)

        path = (
            f"{self.save_dir}/final_model.pt"
            if final else
            f"{self.save_dir}/checkpoint_{update_count}.pt"
        )

        self.mappo.save(path)

        print(f"✓ Saved model -> {path}")

    # --------------------------------------------------------
    # Evaluation + Manual Video Recording
    # --------------------------------------------------------

    def evaluate(self, n_episodes=5, render=True):

        print("\n" + "=" * 70)
        print("Evaluating & Recording")
        print("=" * 70)

        eval_env = WarehouseEnv(
            height=self.env.height,
            width=self.env.width,
            n_robots=self.env.n_robots,
            max_steps=self.env.max_steps,
            render_mode="rgb_array" if render else None
        )

        # ✅ IMPROVED: Manual video recording instead of RecordVideo wrapper
        recorder = None
        if render:
            try:
                from video_recorder import VideoRecorder
                import os
                os.makedirs("videos", exist_ok=True)
                recorder = VideoRecorder("videos/mappo_eval.mp4", fps=10)
                print("🎥 Recording to videos/mappo_eval.mp4")
            except Exception as e:
                print(f"⚠️  Could not initialize video recorder: {e}")
                recorder = None

        rewards_list = []

        for ep in range(n_episodes):

            obs, _ = eval_env.reset()
            done = False
            ep_reward = 0
            step = 0

            while not done:

                global_state = eval_env.get_global_state()
                actions = {}

                for agent in eval_env.agents:
                    action, _, _ = self.mappo.select_action(
                        obs[agent],
                        global_state,
                        deterministic=True
                    )
                    actions[agent] = action

                obs, rewards, terms, truncs, _ = eval_env.step(actions)

                # ✅ FIXED: Render and record frame
                if render and recorder:
                    frame = eval_env.render()
                    if frame is not None:
                        recorder.add_frame(frame)

                ep_reward += np.mean(list(rewards.values()))
                done = any(terms.values()) or any(truncs.values())
                step += 1

            rewards_list.append(ep_reward)
            print(f"Episode {ep+1} | Reward: {ep_reward:.2f} | Steps: {step}")

        if recorder:
            recorder.close()
            print("✓ Video saved!")

        eval_env.close()

        print("\nRESULTS")
        print(f"Mean reward: {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
        print("=" * 70)


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

if __name__ == "__main__":

    CONFIG = {
        "height": 12,
        "width": 12,
        "n_robots": 3,
        "n_shelves": 8,
        "n_drops": 2,
        "max_steps": 500,
        "view_size": 5,

        "total_timesteps": 200_000,
        "n_steps": 2048,
        "n_epochs": 4,
        "batch_size": 64
    }

    print("Creating environment...")
    env = WarehouseEnv(
        height=CONFIG["height"],
        width=CONFIG["width"],
        n_robots=CONFIG["n_robots"],
        n_shelves=CONFIG["n_shelves"],
        n_drops=CONFIG["n_drops"],
        max_steps=CONFIG["max_steps"],
        view_size=CONFIG["view_size"],
        render_mode=None
    )

    obs_sample = env.reset()[0][env.agents[0]]
    global_state_sample = env.get_global_state()

    print("Creating MAPPO...")
    mappo = MAPPO(
        obs_channels=obs_sample.shape[0],
        view_size=obs_sample.shape[1],
        n_actions=7,
        global_state_dim=len(global_state_sample)
    )

    trainer = MAPPOTrainer(
        env,
        mappo,
        n_steps=CONFIG["n_steps"],
        n_epochs=CONFIG["n_epochs"],
        batch_size=CONFIG["batch_size"]
    )

    trainer.train(CONFIG["total_timesteps"])

    # RECORD VIDEO
    trainer.evaluate(n_episodes=3, render=True)