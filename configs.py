MASTER_SEED = 42
FIXED_LAYOUT = True

CONFIG = {
    # Environment
    'height': 12,
    'width': 12,
    'n_robots': 3,
    'n_shelves': 8,
    'n_drops': 2,
    'max_steps': 500,
    'view_size': 5,

    # MAPPO
    'lr_actor': 3e-4,
    'lr_critic': 1e-3,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_coef': 0.5,

    # Training
    'total_timesteps': 1_000_000,
    'n_steps': 2048,
    'n_epochs': 4,
    'batch_size': 64,
    'save_interval': 50,
    'log_interval': 10,
}
