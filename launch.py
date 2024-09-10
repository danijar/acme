import pathlib

from launcher import Runs, launch

repo = pathlib.Path(__file__).parent
gcsdir = 'gs://dm-gdm-worldmodel-team-us-ut-da6189d76684/logdir'
jobs = {
    'agent': {
        'executable': {
            'buildroot': f'{repo}',
            'dockerfile': f'{repo}/Dockerfile',
            'entrypoint': 'python3 examples/baselines/rl_continuous/run_d4pg.py',
        },
        'requirements': {
            'location': 'us-central1',
            'priority': 200,
            'chips': 'a100=1',
        },
        'flags': {
            'logdir': f'{gcsdir}/{{experiment}}/{{task}}/{{method}}/{{seed}}',
        },
    },
}

DMC20 = [
    'dmc_cartpole_swingup', 'dmc_cartpole_balance_sparse', 'dmc_cup_catch',
    'dmc_reacher_hard', 'dmc_finger_turn_hard', 'dmc_cheetah_run',
    'dmc_finger_spin', 'dmc_walker_run', 'dmc_cartpole_swingup_sparse',
    'dmc_hopper_stand', 'dmc_hopper_hop', 'dmc_quadruped_walk',
    'dmc_quadruped_run', 'dmc_pendulum_swingup', 'dmc_acrobot_swingup',
    'dmc_walker_walk', 'dmc_cartpole_balance', 'dmc_finger_turn_easy',
    'dmc_reacher_easy', 'dmc_walker_stand',
]

DMC18 = [
    'dmc_cartpole_swingup', 'dmc_cartpole_balance_sparse', 'dmc_cup_catch',
    'dmc_reacher_hard', 'dmc_finger_turn_hard', 'dmc_cheetah_run',
    'dmc_finger_spin', 'dmc_walker_run', 'dmc_cartpole_swingup_sparse',
    'dmc_hopper_stand', 'dmc_hopper_hop', 'dmc_pendulum_swingup',
    'dmc_acrobot_swingup', 'dmc_walker_walk', 'dmc_cartpole_balance',
    'dmc_finger_turn_easy', 'dmc_reacher_easy', 'dmc_walker_stand',
]
assert all(x in DMC20 for x in DMC18)

runs = Runs({'jobs': jobs})
runs.state = {'obs': 'state'}
runs.times(task=DMC18, seed=range(3))
launch(runs, 'd4pg', alloc='dm/gdm-worldmodels-gcp', tbdir='')
