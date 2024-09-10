# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for rl_continuous experiments."""

from acme import wrappers
import dm_env
import gym


_VALID_TASK_SUITES = ('gym', 'control')


def make_environment(suite, task, logdir) -> dm_env.Environment:
  """Makes the requested continuous control environment.

  Args:
    suite: One of 'gym' or 'control'.
    task: Task to load. If `suite` is 'control', the task must be formatted as
      f'{domain_name}:{task_name}'

  Returns:
    An environment satisfying the dm_env interface expected by Acme agents.
  """

  if suite not in _VALID_TASK_SUITES:
    raise ValueError(
        f'Unsupported suite: {suite}. Expected one of {_VALID_TASK_SUITES}')

  if suite == 'gym':
    env = gym.make(task)
    # Make sure the environment obeys the dm_env.Environment interface.
    env = wrappers.GymWrapper(env)

  elif suite == 'control':
    # Load dm_suite lazily not require Mujoco license when not using it.
    from dm_control import suite as dm_suite  # pylint: disable=g-import-not-at-top
    domain_name, task_name = task.split(':')
    env = dm_suite.load(domain_name, task_name)
    env = wrappers.ConcatObservationWrapper(env)

  # Wrap the environment so the expected continuous action spec is [-1, 1].
  # Note: this is a no-op on 'control' tasks.
  env = wrappers.CanonicalSpecWrapper(env, clip=True)
  env = wrappers.SinglePrecisionWrapper(env)

  env = LoggingWrapper(env, logdir)

  return env


import json
import os
import time

import elements


class LoggingWrapper:

  once = True

  def __init__(self, env, logdir):
    assert type(self).once
    type(self).once = False
    self.env = env
    logdir = elements.Path(logdir)
    logdir.mkdir()
    self.filename = logdir / 'scores.jsonl'
    self.score = 0.0
    self.total = 0
    self.start = time.time()

  @property
  def unwrapped(self):
    return self.env

  def __getattr__(self, name):
    return getattr(self.env, name)

  def reset(self):
    self.score = 0.0
    self.total += 1
    return self.env.reset()

  def step(self, action):
    self.total += 1

    ts = self.env.step(action)

    self.score += float(ts.reward)  # TODO
    if ts.last():
      mins = round((time.time() - self.start) / 60, 1)
      print('episode done!', self.total, mins, self.score, flush=True)
      with self.filename.open('a') as f:
        f.write(json.dumps({'xs': self.total, 'ys': self.score}) + '\n')
      # self.output([(self.total, 'episode/score', self.score)])
      self.score = 0.0

    if self.total >= 1.1e6:
      print('-' * 79)
      print('REACHED ENOUGH STEPS; STOPPING!')
      print('-' * 79)
      os._exit(0)

    return ts
