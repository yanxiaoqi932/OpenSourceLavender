import numpy as np
from typing import List, Tuple


def refer_core(core_config:List[int], unit_scale:int = 1) -> List[str]:   
    """
    translate core config to core command, [2,4,3] => ["0,1","2,3,4,5","6,7,8"]
    input: core config list
    output: core command list
    """
    core_config = [c * unit_scale for c in core_config]
    core_allocation_list = [""] * len(core_config)
    endpoint_left = 0
    for i in range(len(core_config)):
        endpoint_right = endpoint_left + core_config[i] - 1
        core_list = list(range(endpoint_left, endpoint_right+1))
        for j in range(len(core_list)):
            if core_list[j] > 27:
                core_list[j] += 28
        core_allocation_list[i] = ",".join([str(c) for c in core_list])
        endpoint_left = endpoint_right + 1
    return core_allocation_list

def refer_llc(llc_config:List[int], num_llc:int) -> List[str]:
    llc = [-1] * num_llc
    group_id = np.argsort(-np.array(llc_config)).tolist()
    llc_config = (-np.sort(-np.array(llc_config))).tolist()  
    llc_allocation_list_nosorted = []
    for app in range(len(llc_config)):
      ini_list = [0] * num_llc
      size_config = llc_config[app]
      now_config = 0

      while now_config < size_config:
            min_llc = min(llc)
            for i in range(len(llc)):
                  if llc[i] == min_llc and ini_list[i] == 0:
                        ini_list[i] = 1
                        now_config += 1
                        llc[i] += 1
                        if now_config == size_config:
                              break
      llc_allocation_list_nosorted.append(hex(int(''.join([str(item) for item in ini_list]), 2)))        

    llc_allocation_list = []
    for i in range(len(llc_config)):
      for j in range(len(group_id)):
            id = group_id[j]
            if i == id:
                  llc_allocation_list.append(llc_allocation_list_nosorted[j])
    return llc_allocation_list

def refer_mb(mb_config:List[int]) -> List[int]:    
    mb_allocation_list = [i * 10 for i in mb_config]
    return mb_allocation_list

def gen_configs_recursively_fix(num_res:int, num_apps:int) -> List[List[int]]:
    """
    get a resource's allocation space.
    input: resource id, num of groups/apps
    return: a list contains all allocation plans of a resource
    """
    def gen_configs_recursively(u, num_res, a, num_apps):
        if (a == num_apps - 1):
            return None
        else:
            ret = []
            for i in range(1, num_res - u + 1 - num_apps + a + 1):
                confs = gen_configs_recursively(u + i, num_res, a + 1, num_apps)
                if not confs:
                    ret.append([i])
                else:
                    for c in confs:
                        ret.append([i])
                        for j in c:
                            ret[-1].append(j)
            return ret
    res_config = gen_configs_recursively(0, num_res, 0, num_apps)
    for i in range(len(res_config)):
        other_source = np.array(res_config[i]).sum()
        res_config[i].append(num_res - other_source)
    return res_config


def gen_init_config(app_num:int=0, num_core:int=0, num_llc:int=0, num_mb:int=0,
                    core_space:List[List[int]]=[], llc_space:List[List[int]]=[], mb_space:List[List[int]]=[]):
    """
    create an initial resource plan.
    input: app number, core, llc and mb number
    return: all resources' config, and corrsponding arms
    """
    nof_core = num_core; nof_llc = num_llc; nof_mb = num_mb
    core_arm = 0; llc_arm = 0; mb_arm = 0
    
    core_config = split_averagely(nof_units=nof_core, nof_clusters=app_num)
    for config_id in range(len(core_space)):
        if core_config == core_space[config_id]:
            core_arm = config_id
            break

    if nof_llc != 0:
        llc_config = split_averagely(nof_units=nof_llc, nof_clusters=app_num)
    for config_id in range(len(llc_space)):
        if llc_config == llc_space[config_id]:
            llc_arm = config_id
            break

    if nof_mb != 0:
        mb_config = split_averagely(nof_units=nof_mb, nof_clusters=app_num)
    for config_id in range(len(mb_space)):
        if mb_config == mb_space[config_id]:
            mb_arm = config_id
            break

    return [core_config, llc_config, mb_config]

def perform_resource_partitioning(config, group_list, app_list):
     # Use taskset, cat and mba to allocate resources to jobs
    ipc_list = allocate_resource(config, group_list, app_list)
    
def get_now_ipc(app_list, core_config):
    ipc_list, th = [], []
    for i in range(len(app_list)):
        # Use Perf tool to get the values of all jobs' selected PMCs
        i, t = perf(app_list[i], core_config[i])
        ipc_list.append(i); th.append(t)
    return th, ipc_list

def LatinSample(core_space, llc_space=0, mb_space=0):
    return sample(core_space, llc_space, mb_space)
    
def get_best_config(rewards):
    return np.argmax(rewards) 

def split_averagely(nof_units:int, nof_clusters:int) -> List[int]:
    each_clu_units = nof_units // nof_clusters
    res_clu_units = nof_units % nof_clusters
    units_clu = [each_clu_units] * (nof_clusters-1)
    if res_clu_units >= each_clu_units:
        for i in range(res_clu_units):
            units_clu[i]+=1
        units_clu.append(each_clu_units)
    else:
        units_clu.append(each_clu_units+res_clu_units)
    return units_clu