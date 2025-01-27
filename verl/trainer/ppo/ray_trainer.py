# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
import ragen
from ragen.utils import set_seed

import shutil

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # averagen over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 env=None,
                 env_class=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.env = env
        self.env_class = env_class

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)
    
    def _convert_pad_structure(self, tensor, pad_token_id, pad_to_left=True):
        """originally: [pad_left, content, pad_right]
        want to: [pad_right, pad_left, content] / [content, pad_right, pad_left]
        """
        if pad_to_left:
            mask = tensor != pad_token_id
        else:
            mask = tensor == pad_token_id

        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        converted_tensor = tensor.gather(1, sorted_indices)

        return converted_tensor, sorted_indices

    def _cut_to_effective_len(self, protocol, keys, cut_left=True):
        assert 'attention_mask' in protocol.batch, 'attention_mask is required'
        effective_len = protocol.batch['attention_mask'].sum(dim=1).max()
        for key in keys:
            if cut_left:
                protocol.batch[key] = protocol.batch[key][:, -effective_len:]
            else:
                protocol.batch[key] = protocol.batch[key][:, :effective_len]
        return protocol

    def _batch_tokenize(self, responses):
        return self.tokenizer(responses, add_special_tokens=False, return_tensors='pt', padding="longest")['input_ids']

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        K = self.config.max_turns
        max_start_len = self.config.data.max_start_length # the first round prompt max len
        max_prompt_len = self.config.data.max_prompt_length # the max len of prompt
        max_response_len = self.config.data.max_response_length # the max length of response
        max_seq_len = max_prompt_len + max_response_len
        max_obs_len = self.config.data.max_obs_length

        dim_room_x, dim_room_y = self.config.env.dim_x, self.config.env.dim_y
        num_boxes = self.config.env.num_boxes
        max_steps = self.config.env.max_steps
        search_depth = self.config.env.search_depth
        envs = [self.env.copy() for _ in range(self.config.data.train_batch_size)]

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                env_seeds = [i['index'] for i in batch.non_tensor_batch['extra_info']]
                print("env_seeds:", env_seeds)
                for env, seed in zip(envs, env_seeds):
                    env.reset(seed=seed)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                # with _timer('gen', timing_raw):
                #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                #     batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                #                                              dtype=object)
                #     # repeat to align with repeated responses in rollout
                #     batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                #     batch = batch.union(gen_batch_output)

                #     # output batch to file
                #     self._record_batch(batch, path=f'.log/{self.config.trainer.experiment_name}/gen_batch.txt')

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################

                # keep the very first input_ids
                first_input_ids = gen_batch.batch['input_ids'][:, -max_start_len:].clone()

                with _timer('step', timing_raw):
                    """
                    keep rolling to generate K turns of responses.
                    when doing this, update the "original right side" when new responses are generated.
                    finally, concatenate the "original left side" and "original right side" to get the final thing to feed to train the model.

                    Left-pad prompts, right-gen flow, Tensors dance like stardust glow.
                    Errors swarm? Stay calm, don't fret- Code with coffee, debug sunset.
                    """
                
                    rollings = gen_batch
                                        
                    original_left_side = {
                        'input_ids': gen_batch.batch['input_ids'][:, -max_start_len:].clone(), 
                    }
                    original_right_side = {
                        'responses': gen_batch.batch['input_ids'][:, []].clone(),
                        ## 'old_log_probs': gen_batch.batch['input_ids'][:, []].clone(),
                    }
                    meta_info = {}
                    
                    # # if exists, remove the existing log
                    # if os.path.exists(f'.log/{self.config.trainer.experiment_name}'):
                    #     shutil.rmtree(f'.log/{self.config.trainer.experiment_name}')

                    for rollout_step in range(K):
                        with _timer(f'gen', timing_raw):
                            # cut to effective length
                            rollings = self._cut_to_effective_len(rollings, keys=['input_ids', 'attention_mask', 'position_ids'], cut_left=True)
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(rollings)
                            meta_info.update(gen_batch_output.meta_info)

                        # process gen_batch_output, remove all things like reward: xxx \n to forbid reward hacking
                        # decode responses, remove all things like reward: xxx \n to forbid reward hacking
                        cur_responses_decoded = self.tokenizer.batch_decode(gen_batch_output.batch['responses'], skip_special_tokens=False)
                        # ifthere has been hacks, output this to a log 
                        hack_pattern = r'reward: \d+\.\d+\n|done: (True|False)\n'
                        hacked_responses = [response for response in cur_responses_decoded if re.search(hack_pattern, response)]
                        if len(hacked_responses) > 0:
                            print(f"[WARNING] HACKED RESPONSES: {hacked_responses}")
                            cur_responses_decoded = [re.sub(hack_pattern, '', response) for response in cur_responses_decoded]
                            # see if there is any hack left
                            if len(hacked_responses) == 0:
                                print(f"[DEBUG] No hack left in responses.")
                            gen_batch_output.batch['responses'] = self._batch_tokenize(cur_responses_decoded)

                        os.makedirs(f'.log/{self.config.trainer.experiment_name}/rollout_step_{rollout_step}', exist_ok=True)
                        import datetime
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(f'.log/{self.config.trainer.experiment_name}/rollout_step_{rollout_step}/left_side.txt', 'w') as f:
                            f.write(f"{now}\n")
                            f.write(f"[left side]: \n{rollings}\n")
                            f.write(f"[left side shape]: \n{rollings.batch['input_ids'].shape}\n")
                            for idx in range(4):
                                f.write(f"[left side decoded]: \n{self.tokenizer.decode(rollings.batch['input_ids'][idx], skip_special_tokens=False)}\n")
                            f.write(f"\n")

                        with open(f'.log/{self.config.trainer.experiment_name}/rollout_step_{rollout_step}/right_side.txt', 'w') as f:
                            f.write(f"{now}\n")
                            f.write(f"[right side]: \n{gen_batch_output}\n")
                            f.write(f"[right side shape]: \n{gen_batch_output.batch['responses'].shape}\n")
                            for idx in range(4):
                                f.write(f"[right side decoded]: \n{self.tokenizer.decode(gen_batch_output.batch['responses'][idx], skip_special_tokens=False)}\n")
                            f.write(f"\n")


                        # # Update here to plot the trajectory and thought
                        # import matplotlib.pyplot as plt
                        # from matplotlib.backends.backend_pdf import PdfPages
                        # from matplotlib.gridspec import GridSpec


                        # def save_trajectory_to_pdf(trajectory, filename='trajectory_visualization.pdf'):
                        #     with PdfPages(filename) as pdf:
                        #         for batch_idx, data in enumerate(trajectory):


                        #             # fig, ax = plt.subplots(2, 2, figsize=(10, 8))
                                    
                        #             # # Plot before action image (top left)
                        #             # ax[0, 0].imshow(data.get('img_before_action'))
                        #             # ax[0, 0].set_title('Before Action')
                        #             # ax[0, 0].axis('off')
                                    
                        #             # # Plot after action image (bottom left)
                        #             # ax[1, 0].imshow(data.get('img_after_action'))
                        #             # ax[1, 0].set_title('After Action')
                        #             # ax[1, 0].axis('off')

                        #             # # Display thought (top right)
                        #             # ax[0, 1].text(0.5, 0.5, f"Thought:\n{data.get('thought', '')}",
                        #             #             ha='center', va='center', fontsize=12, wrap=True)
                        #             # ax[0, 1].set_title('Thought')
                        #             # ax[0, 1].axis('off')
                                    
                        #             # # Display action (bottom right)
                        #             # ax[1, 1].text(0.5, 0.5, f"Action:\n{data.get('action', '')}",
                        #             #             ha='center', va='center', fontsize=12, wrap=True)
                        #             # ax[1, 1].set_title('Action')
                        #             # ax[1, 1].axis('off')

                        #             # Create figure and custom gridspec layout
                        #             fig = plt.figure(figsize=(10, 8))
                        #             gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], figure=fig)

                        #             # Plot before action image (top left)
                        #             ax1 = fig.add_subplot(gs[0, 0])
                        #             ax1.imshow(data.get('img_before_action'))
                        #             ax1.set_title('Before Action')
                        #             ax1.axis('off')

                        #             # Plot after action image (bottom left)
                        #             ax2 = fig.add_subplot(gs[1, 0])
                        #             ax2.imshow(data.get('img_after_action'))
                        #             ax2.set_title('After Action')
                        #             ax2.axis('off')

                        #             # Display answer (right side)
                        #             ax3 = fig.add_subplot(gs[:, 1])
                        #             ax3.text(0.5, 0.5, f"Answer:\n{data.get('answer', '')}",
                        #                     ha='center', va='center', fontsize=12, wrap=True)
                        #             ax3.set_title('Answer')
                        #             ax3.axis('off')

                                    
                                    
                        #             plt.suptitle(f'Batch {batch_idx + 1}', fontsize=16)
                        #             pdf.savefig(fig)
                        #             plt.close(fig)
                        #     print(f'PDF saved as {filename}')
                        # trajectory = []
                        # for env in envs[:4]:
                        #     trajectory.append({
                        #         "img_before_action": env.render('rgb_array'),
                        #     })
                        with _timer('execute_predictions', timing_raw):
                            cur_responses_decoded = self.tokenizer.batch_decode(gen_batch_output.batch['responses'], skip_special_tokens=False)
                            next_obs = self.env_class.execute_predictions(envs, cur_responses_decoded, self.tokenizer.pad_token)
                        
                        # for idx, (cur_response_decoded, env) in enumerate(zip(cur_responses_decoded[:4], envs[:4])):
                        #     img_after_action = env.render('rgb_array')
                        #     # print(f"[CUR_RESPONSE_DECODED]: {cur_response_decoded} [\CUR_RESPONSE_DECODED]")
                        #     # if '<answer>' in cur_response_decoded:
                        #     #     action = cur_response_decoded.split("<answer>")[1].split("</answer>")[0].strip()
                        #     # else:
                        #     #     action = ""
                        #     # if "<think>" in cur_response_decoded:
                        #     #     thought = cur_response_decoded.split("<think>")[1].split("</think>")[0].strip()
                        #     # else:
                        #     #     thought = ""
                        #     trajectory[idx].update({
                        #         "img_after_action": img_after_action,
                        #         "answer": cur_response_decoded,
                        #         # "action": action,
                        #         # "thought": thought,
                        #     })
                        # save_trajectory_to_pdf(trajectory, filename=f'.log.debug/rollout_step_{rollout_step}/trajectory_visualization.pdf')

                        




                        next_obs_input_ids = self.tokenizer(next_obs, padding='longest', return_tensors='pt')['input_ids']
                        if next_obs_input_ids.shape[1] > max_obs_len:
                            print("[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG")
                            print(next_obs_input_ids.shape)
                            next_obs_input_ids = next_obs_input_ids[:, :max_obs_len]

                        cur_responses = gen_batch_output.batch['responses']          # [BSZ, MAX_RESP_LEN]
                        ## cur_log_probs = gen_batch_output.batch['old_log_probs']

                        # add responses to the rollings
                        new_input_ids = torch.cat([
                            rollings.batch['input_ids'], 
                            cur_responses,
                            next_obs_input_ids
                        ], dim=1)

                        # update rollings, pad to the left
                        new_input_ids, _ = self._convert_pad_structure(new_input_ids, pad_token_id=self.tokenizer.pad_token_id, pad_to_left=True)
                        new_attention_mask = torch.where(new_input_ids != self.tokenizer.pad_token_id, 1, 0)
                        new_position_ids = (torch.cumsum(new_attention_mask, dim=1) - 1) * new_attention_mask

                        effective_len = new_attention_mask.sum(dim=1).max()
                        max_len = min(max_prompt_len, effective_len)

                        # cut to the min of max_prompt_len and effective_len
                        new_input_ids = new_input_ids[:, -max_len:]
                        new_attention_mask = new_attention_mask[:, -max_len:]
                        new_position_ids = new_position_ids[:, -max_len:]

                        rollings = DataProto.from_dict({
                            'input_ids': new_input_ids,
                            'position_ids': new_position_ids,
                            'attention_mask': new_attention_mask
                        })


                        # update original right side, pad to the right
                        original_right_side['responses'] = torch.cat([
                            original_right_side['responses'],
                            cur_responses,
                            next_obs_input_ids
                        ], dim=1)
                        
                        original_right_side['responses'], indices = self._convert_pad_structure(original_right_side['responses'], pad_token_id=self.tokenizer.pad_token_id, pad_to_left=False)
                        ## original_right_side['old_log_probs'] = torch.gather(original_right_side['old_log_probs'], 1, indices)
                        
                        # also make a cut
                        effective_len = torch.where(original_right_side['responses'] != self.tokenizer.pad_token_id, 1, 0).sum(dim=1).max()
                        max_len = min(max_prompt_len, effective_len)
                        original_right_side['responses'] = original_right_side['responses'][:, :max_len]


                    # compose final gen batch output
                    with _timer('compose_final_gen_batch_output', timing_raw):
                        final_gen_batch_output = original_right_side
                        final_gen_batch_output['prompts'] = original_left_side['input_ids']
                        final_gen_batch_output['input_ids'] = torch.concat([
                            original_left_side['input_ids'],
                            original_right_side['responses']
                        ], dim=1)
                        final_gen_batch_output['attention_mask'] = torch.concat([
                            torch.where(original_left_side['input_ids'] != self.tokenizer.pad_token_id, 1, 0),
                            torch.where(final_gen_batch_output['responses'] != self.tokenizer.pad_token_id, 1, 0),
                        ], dim=1)
                        final_gen_batch_output['position_ids'] = (torch.cumsum(final_gen_batch_output['attention_mask'], dim=1) - 1) * final_gen_batch_output['attention_mask']

                        final_gen_batch_output = DataProto.from_dict(original_right_side)
                        final_gen_batch_output.meta_info.update(meta_info)

                    # to get the old_log_probs
                    with _timer('ref_probs_forward', timing_raw):
                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)  # [IMPLEMENT]
                            final_gen_batch_output = final_gen_batch_output.union(output)
                            # print output length
                            print(f"everything about output like length: {output}")
                        # response_len = original_right_side['responses'].shape[1]
                    #     final_gen_batch_output['old_log_probs'] = output[:, -response_len:]

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################


                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
