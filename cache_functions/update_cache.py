import torch
def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    '''
    Update the cache with the fresh tokens.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']

    indices = fresh_indices

    cache_dic['cache'][-1][layer][module][0].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)
    
    # updated_cache = {}
    # updated_cache[0] = fresh_tokens
    # if fresh_tokens.shape[1] == 0:
    #     return
    # difference_distance = torch.full_like(current['activated_steps'][layer][module], step, device=fresh_tokens.device) - current['activated_steps'][layer][module]
    # fresh_difference_distance = difference_distance.gather(dim=1, index=indices)
    
    # for i in range(cache_dic['max_order']):
    #     if cache_dic['cache'][-1][layer][module].get(i, None) is not None:
    #         updated_cache[i + 1] = (updated_cache[i] - cache_dic['cache'][-1][layer][module][i].gather(1, indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]))) / fresh_difference_distance.unsqueeze(-1)
    #     else:
    #         break
    # for i in range(cache_dic['max_order']):
    #     cache_dic['cache'][-1][layer][module][i].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=updated_cache[i])
        

def smooth_update_cache(fresh_indices, fresh_tokens, cache_dic, current):
    step = current['step']
    layer = current['layer']
    module = current['module']

    cluster_indices, cluster_nums, topk = \
        cache_dic['cluster_info']['cluster_indices'], cache_dic['cluster_info']['cluster_nums'], cache_dic['cluster_info']['topk']
    smooth_rate = cache_dic['smooth_rate']
    dim = fresh_tokens.shape[-1]
    cache_dic['cache'][-1][layer][module][0].scatter_(dim=1, index=fresh_indices.unsqueeze(-1).expand(-1, -1, dim), src=fresh_tokens)
    old_cache = cache_dic['cache'][-1][layer][module][0]
    B, N, dim = old_cache.shape
    device = old_cache.device

    fresh_cluster_indices = cluster_indices.gather(dim=1, index=fresh_indices)

    sum_per_cluster = torch.zeros((B, cluster_nums, dim), device=device)
    sum_per_cluster.scatter_(
        dim=1,
        index=fresh_cluster_indices.unsqueeze(-1).expand(-1, -1, dim),
        src=fresh_tokens.float()
    )

    mean_per_cluster = sum_per_cluster                                                       # only when topk == 1
    # mean_per_cluster = sum_per_cluster / count_per_cluster.unsqueeze(-1).clamp(min=1e-6)   # when topk > 1

    new_cache = mean_per_cluster.gather(1, cluster_indices.unsqueeze(-1).expand(-1, -1, dim))

    cache_dic['cache'][-1][layer][module][0] = new_cache * smooth_rate + old_cache * (1 - smooth_rate)