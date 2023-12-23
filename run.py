import subprocess
import os
import sys
import numpy as np
from typing import List
from copy import deepcopy

DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIRNAME))

from utils import gen_configs_recursively_fix, gen_init_config,\
    perform_resource_partitioning, get_now_ipc, LatinSample, get_best_config


def inter_group_ini_opt(group_list:List[List[str]], rounds:int, app_list:List[str]):
    rewards = []; ipc_lists = []; configs = []
    num_core_units = int(NUM_CORE / CORE_UNIT_SCALE)  
    core_space = gen_configs_recursively_fix(num_res=num_core_units, num_apps=NUM_GROUPS)
    llc_space = gen_configs_recursively_fix(num_res=NUM_LLC, num_apps=NUM_GROUPS)
    mb_space = gen_configs_recursively_fix(num_res=NUM_MB, num_apps=NUM_GROUPS)

    config = gen_init_config(app_num=NUM_GROUPS, num_core=num_core_units, num_llc=NUM_LLC, num_mb=NUM_MB,
                    core_space=core_space, llc_space=llc_space, mb_space=mb_space)  
    
    for now_round in range(rounds):
        print(f"Start run {now_round}th round")
        perform_resource_partitioning(config,group_list,app_list)
        reward, ipc_list = get_now_ipc(app_list, config[0])
        print(f'**********{now_round}th round, throughput:{reward}**********')
        
        rewards.append(reward); ipc_lists.append(ipc_list)
        configs.append(config)
        config = LatinSample(core_space=core_space, llc_space=llc_space, mb_space=mb_space)
    best_config, best_th, best_ipc_list = get_best_config(rewards, ipc_list)
    return best_config, best_th, best_ipc_list

def inter_group_gra_opt(best_config, best_th, best_ipc_list, rounds,
                        group_list, app_list):
    # state = # <group_id，resource type，upper bound(0)/lower bound(0)，number of resource>
    state_table = np.ones(shape=(NUM_GROUPS, NUM_RESOURCES, 2, 1))
    for a in range(NUM_GROUPS):
        for b in range(NUM_RESOURCES):
            state_table[a, b, 0] = 10000
            state_table[a, b, 1] = -10000

    config = [best_config[0], best_config[1], best_config[2]]
    b_ipc_list = best_ipc_list
    b_th = best_th

    for now_round in range(rounds):
        res_id = now_round % NUM_RESOURCES
        sorted_group_id = np.argsort(np.array(b_ipc_list)) 
        h_group_id = -1; l_group_id = -1
        for i in range(NUM_GROUPS-1, -1, -1):
            g = sorted_group_id[i]
            if config[res_id][g] > 1 and config[res_id][g] > state_table[g, res_id, 1]:
                h_group_id = g; break
        for i in range(NUM_GROUPS):
            g = sorted_group_id[i]
            if config[res_id][g] < state_table[g, res_id, 0]:
                if (res_id == 0) or (res_id == 1 and config[res_id][g] < NUM_LLC) or (res_id == 2 and config[res_id][g] < NUM_MB):
                    l_group_id = g; break
        if h_group_id != -1 and l_group_id != -1:
            config[res_id][h_group_id] -= 1; config[res_id][l_group_id] += 1
            have_changed = True
        else:
            have_changed = False
        
        perform_resource_partitioning(config, group_list=group_list, app_list=app_list, is_group=True)
        new_reward, new_ipc_list = get_now_ipc(app_list, config)
        print(f'\n**********{now_round}th round, throughput:{new_reward}**********\n')
        
        if have_changed:
            if new_ipc_list[h_group_id] < (1 - PERCENT_DECSEND) * b_ipc_list[h_group_id]:
                state_table[h_group_id, res_id, 1] = config[res_id][h_group_id] + 1
            if new_ipc_list[l_group_id] < (1 + PERCENT_RAISE) * b_ipc_list[l_group_id]:
                state_table[l_group_id, res_id, 0] = config[res_id][l_group_id] - 1
            if new_reward < b_th:
                config[res_id][h_group_id] += 1; config[res_id][l_group_id] -= 1
            else:
                b_ipc_list = new_ipc_list
                b_th = new_reward
        else:
            b_ipc_list = new_ipc_list
            b_th = new_reward
    return config, b_th, b_ipc_list

def intra_group_ini_opt(group_list, rounds, app_list, g_core_config, llc_config, mb_config):
    rewards = []; ipc_lists = []; configs = []
    g_core_config = [c * CORE_UNIT_SCALE for c in g_core_config]
    all_core_space = []
    nonsort_all_core_config = []
    for g_id in range(len(group_list)): 
        num_apps = len(group_list[g_id])
        core_space = gen_configs_recursively_fix(num_res=g_core_config[g_id], num_apps=num_apps)
        all_core_space.append(core_space)
        core_config = \
            gen_init_config(app_num=num_apps, num_core=g_core_config[g_id], core_space=core_space)
        nonsort_all_core_config.append(core_config)
    for now_round in range(rounds):
        perform_resource_partitioning([core_config, llc_config, mb_config],
                                      group_list=group_list,
                                      app_list=app_list, is_group=False)
        r, ipc_list = get_now_ipc(app_list=app_list, core_allocation_list=all_core_allocation_list)
        print(f'**********{now_round}th round, reward:{r}**********')
        rewards.append(r); ipc_lists.append(ipc_list); configs.append(core_config)
        all_core_allocation_list = []
        for g in range(len(group_list)):
            core_config = LatinSample(core_space=all_core_space[g])[0]
            all_core_allocation_list.append(core_config)
        
    best_config, best_th, best_ipc_list = get_best_config(rewards, ipc_list)
    return best_config, best_th, best_ipc_list

def intra_group_gra_opt(best_config, best_th, best_ipc_list, rounds:int,
                        group_list, app_list, llc_config, mb_config):
    
    best_core_config = []; best_core_reward = []; best_core_ipc_list = []
        
    all_core_config = [best_config[g] for g in range(len(group_list))]
    all_b_ipc_list = [best_ipc_list[g] for g in range(len(group_list))]
    all_b_reward = [best_th[g] for g in range(len(group_list))]

    range_config = []; range_reward = []; range_ipc_list = []
    state_table = np.ones(shape=(NUM_GROUPS, NUM_APPS, 2, 1))
    for a in range(NUM_GROUPS):
        for b in range(NUM_APPS):
            state_table[a, b, 0] = 10000
            state_table[a, b, 1] = -10000

    for now_round in range(rounds):
        all_h_app_id = []; all_l_app_id = []
        have_changed = [False] * len(group_list)
        for g in range(len(group_list)):
            sorted_app_id = np.argsort(np.array(all_b_ipc_list[g]))  # 从小到大
            h_app_id = -1; l_app_id = -1
            for i in range(len(sorted_app_id)-1, -1, -1):
                a = sorted_app_id[i]
                if all_core_config[g][a] > 1 and all_core_config[g][a] > state_table[g, a, 1]:
                    h_app_id = a; break
            for i in range(len(sorted_app_id)):
                a = sorted_app_id[i]
                if all_core_config[g][a] < state_table[g, a, 0]:
                    l_app_id = a; break
            
            if h_app_id != -1 and l_app_id != -1:
                have_changed[g] = True
                all_core_config[g][h_app_id] -= 1; all_core_config[g][l_app_id] += 1
            else:
                have_changed[g] = False
            all_h_app_id.append(h_app_id); all_l_app_id.append(l_app_id)

        perform_resource_partitioning([all_core_config, llc_config, mb_config], group_list=group_list, app_list=app_list)
        new_reward, new_ipc_list = get_now_ipc(app_list, all_core_config)
        range_config.append(all_core_config); range_reward.append(new_reward); range_ipc_list.append(new_ipc_list)
        print(f'**********{now_round}th round, reward:{new_reward}**********')
        
        for g in range(len(group_list)):
            h_app_id = all_h_app_id[g]; l_app_id = all_l_app_id[g]
            if have_changed[g]:
                if new_ipc_list[g][h_app_id] < PERCENT_DECSEND * all_b_ipc_list[g][h_app_id]:
                    state_table[g, h_app_id, 1] = all_core_config[g][h_app_id] + 1
                if new_ipc_list[g][l_app_id] < (1 + PERCENT_RAISE) * all_b_ipc_list[g][l_app_id]:
                    state_table[g, l_app_id, 0] = all_core_config[g][l_app_id] - 1
                if new_reward[g] < all_b_reward[g]:
                    all_core_config[g][h_app_id] += 1; all_core_config[g][l_app_id] -= 1
                else:
                    all_b_ipc_list[g] = new_ipc_list[g]
                    all_b_reward[g] = new_reward[g]
            else:
                all_b_ipc_list[g] = new_ipc_list[g]
                all_b_reward[g] = new_reward[g]
    
    best_id = np.argmax(np.array(range_reward))
    best_core_config.append(deepcopy(range_config[best_id]))
    best_core_reward.append(deepcopy(range_reward[best_id]))
    best_core_ipc_list.append(deepcopy(range_ipc_list[best_id]))
    return best_core_config, best_core_reward, best_core_ipc_list


def run(app_list:List[str], group_list:List[List[str]], g_1_rounds:int, g_2_rounds:int, 
          a_1_rounds:int, a_2_rounds:int):
    best_config, best_th, best_ipc_list = \
        inter_group_ini_opt(group_list=group_list, rounds=g_1_rounds, app_list=app_list)
    config, b_th, b_ipc_list = \
        inter_group_gra_opt(best_config, best_th, best_ipc_list,
                            rounds=g_2_rounds, group_list=group_list, app_list=app_list)
    

    best_config, best_th, best_ipc_list = \
        intra_group_ini_opt(group_list, a_1_rounds, app_list, config[0], config[1], config[2])
    best_core_config, best_core_reward, best_core_ipc_list = \
        intra_group_gra_opt(best_config, best_th, best_ipc_list, 
                            a_2_rounds, group_list, app_list, config[1], config[2])
    best_config = [best_core_config, config[0], config[1]]
    return best_config

def monitor(app_list, group_list, config, ini_reward):
    while True:
        perform_resource_partitioning(config, app_list=app_list, group_list=group_list, is_group=False)
        new_reward, new_ipc_list = get_now_ipc(app_list, config[0])
        
        intra_times = 0; inter_times = 0
        if new_reward >= ini_reward * PERCENT_GRA_INTRA:
            intra_times = 0; inter_times = 0
        elif new_reward < ini_reward * PERCENT_GRA_INTRA and new_reward >= ini_reward * PERCENT_GRA_INTER:
            intra_times += 1
            if inter_times != 0: inter_times += 1
        elif new_reward < ini_reward * PERCENT_GRA_INTER:
            inter_times += 1

        if inter_times >= START_GRA_ROUNDS:
            run(app_list, group_list, G_1_ROUNDS, G_2_ROUNDS, A_1_ROUNDS, A_2_ROUNDS)
            inter_times = 0
        elif intra_times >= START_GRA_ROUNDS:
            intra_group_gra_opt(config, new_reward, new_ipc_list, group_list, app_list)
            intra_times = 0


if __name__ == "__main__":
    subprocess.run('sudo pqos -R', shell=True, capture_output=True)
    start = 0.0

    colocation_list, app_list, group_list = [], [], []  # put in colocated jobs
    NUM_APPS = len(colocation_list) 
    NUM_GROUPS = 6

    NUM_CORE = 60  
    NUM_LLC = 10
    NUM_MB = 10
    NUM_UNITS = [NUM_CORE, NUM_LLC, NUM_MB]
    NUM_RESOURCES = 3
    G_1_ROUNDS = 20 
    G_2_ROUNDS = 10 
    A_1_ROUNDS = 10 
    A_2_ROUNDS = 10 

    CORE_UNIT_SCALE = 5 
    PERCENT_DECSEND = 0.3   
    PERCENT_RAISE = 0.1     
    PERCENT_GRA_INTRA = 0.85   
    PERCENT_GRA_INTER = 0.7 
    START_GRA_ROUNDS = 3

    best_config, ini_reward = run(app_list, group_list, G_1_ROUNDS, G_2_ROUNDS, A_1_ROUNDS, A_2_ROUNDS)
    monitor(app_list, group_list, best_config, ini_reward)

