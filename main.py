
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition
from stable_baselines3.common.callbacks import CheckpointCallback
from rlgym.utils.reward_functions.combined_reward import CombinedReward
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from rlgym.make import make
from rlgym.utils.reward_functions.common_rewards import ConstantReward
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards import GoalLineAgentBall
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.wrappers.sb3_wrappers import SB3MultipleInstanceWrapper, SB3VecMonitor
from rlgym.utils.reward_functions.common_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards import RewardIfBehindBall
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards import SaveBoostReward
from rlgym.utils.reward_functions.common_rewards import FaceBallReward
from rlgym.utils.reward_functions.common_rewards import GoalReward
from rlgym.utils.reward_functions.common_rewards import MoveTowardsBallReward
from rlgym.utils.reward_functions.common_rewards import LineUpBallReward
from rlgym.utils.reward_functions.common_rewards import BoostReward
from rlgym.utils.reward_functions.common_rewards import AerialTouchReward
import rlgym.make
from datetime import datetime

if __name__ == '__main__':
    # set episode timeout (1800 sec = 30 min)
    default_tick_skip = 8
    physics_ticks_per_second = 120
    ep_len_seconds = 45
    max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

    # specify terminal conditions from commons
    term_cond1 = TimeoutCondition(max_steps)
    term_cond2 = GoalScoredCondition()

    # specify reward functions
    fps = 15
    obs_builder = AdvancedObs()

    all_rewards = CombinedReward(
        (EventReward(goal=100, team_goal=0, concede=-100, touch=0, shot=10, save=25, demo=10),
         MoveTowardsBallReward(),
         SaveBoostReward(),
         FaceBallReward(),
         GoalReward(),
         AerialTouchReward(),
         BoostReward(),
         LineUpBallReward()),
        (1,
         (.1 / fps),
         (.2 / fps),
         (.1 / fps),
         .25,
         1.5,
         (.2 / fps),
         (.1 / fps)))


    def get_match_args():
        return dict(
            team_size=1,
            tick_skip=8,
            reward_function=all_rewards,
            self_play=True,
            game_speed=100,
            terminal_conditions=[term_cond1, term_cond2],
            obs_builder=obs_builder)


    epic_path = r"C:\\Program Files\\Epic Games\\rocketleague\\Binaries\\Win64\\RocketLeague.exe"
    monitor_path = 'C:\\Users\\jules\\Documents\\SMU\\code2\\reward_monitors\\reward_record_'
    num_procs = 27
    num_ts = 170_000_000
    env = SB3VecMonitor(SB3MultipleInstanceWrapper(epic_path, num_procs, get_match_args, wait_time=15),
                        filename=monitor_path)

    checkpoint = CheckpointCallback(save_freq=1_000_000 // env.num_envs + 1,
                                    save_path='C:\\Users\\jules\\Documents\\SMU\\code2\\PPO_r_u\\',
                                    name_prefix="ppo_bot_",
                                    verbose=1)
    
    # learner = PPO("MlpPolicy", env, verbose=3, n_epochs=1, target_kl=0.02 / 1.5)

    learner = PPO.load(path='C:\\Users\\jules\\Documents\\SMU\\code2\\PPO2_r_u\\ppo_bot__120003120_steps',
                       env=env,
                       n_epochs=1,
                       target_kl=0.02 / 1.5,
                       verbose=3,
                       learning_rate=3e-4)


    learner.learn(total_timesteps=num_ts,
                  callback=checkpoint)
                  
    learner.save('C:\\Users\\jules\\Documents\\SMU\\code2\\PPO2_r_u_backup\\ppo_bot__290_000_000')

