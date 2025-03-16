def cache_init(model_kwargs, num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache[-1]={}

    for j in range(28):
        cache[-1][j] = {}
    for i in range(num_steps):
        cache[i]={}
        for j in range(28):
            cache[i][j] = {}

    cache_dic['cache']                = cache
    cache_dic['flops']                = 0.0
    cache_dic['interval']             = model_kwargs['interval']
    cache_dic['max_order']            = model_kwargs['max_order']
    cache_dic['test_FLOPs']           = model_kwargs['test_FLOPs']
    cache_dic['first_enhance']        = 2
    cache_dic['cache_counter']        = 0

    if (model_kwargs.get('fresh_ratio', None) is None) or (model_kwargs['fresh_ratio'] == 0.0):
    
        cache_dic['enable_toca']          = False
    
    else:
        cache_dic['enable_toca']          = True
        cache_dic['fresh_ratio_schedule'] = model_kwargs['ratio_scheduler']
        cache_dic['fresh_ratio']          = model_kwargs['fresh_ratio']
        cache_dic['soft_fresh_weight']    = model_kwargs['soft_fresh_weight']

        cache_index = {}
        cache_index[-1]={}
        for j in range(28):
            cache[-1][j] = {}
            cache_index[-1][j] = {}
    
        cache_dic['cache_index']          = cache_index
        
    current = {}
    current['num_steps'] = num_steps
    current['activated_steps'] = [49]
    return cache_dic, current
    