import pathlib

from launcher import Runs, launch

def jobs(agent='d4pg'):
  repo = pathlib.Path(__file__).parent
  gcsdir = 'gs://dm-gdm-worldmodel-team-us-ut-da6189d76684/logdir'
  rundir = '{random}-{task}-{method}-{seed}'
  jobs = {
      'agent': {
          'executable': {
              'buildroot': f'{repo}',
              'dockerfile': f'{repo}/Dockerfile',
              'entrypoint': (
                  'python3 examples/baselines/rl_continuous/run_{agent}.py'),
          },
          'requirements': {
              'location': 'us-central1',
              'priority': 200,
              'chips': 'a100=1',
          },
          'flags': {
              'logdir': f'{gcsdir}/{{experiment}}/{rundir}',
          },
      },
  }
  return jobs

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

DMC20 = [
    ('control:' + x[4:].replace('_', ':', 1)).replace(':cup:', ':ball_in_cup:')
    for x in DMC20]

# print('\n'.join(DMC20))
# import sys; sys.exit()

runs = Runs()
runs.d4pg = {'jobs': jobs('d4pg')}
runs.dmpo = {'jobs': jobs('dmpo')}
runs.ppo = {'jobs': jobs('ppo')}
runs.times(task=DMC20, seed=range(3))
launch(runs, 'acme', alloc='dm/gdm-worldmodels-gcp', tbdir='')

