#%%
import util
#%%
base_dir = '/mnt/LEB/Scientific_projects/deep_events_WS/data/single_channel_fluo/MitoSplit-Net/'
data_path = base_dir+'Data/' 
model_path = base_dir+'Models/'


#%% 
def load_best_score(name: str):
    ref_f1 = util.load_pkl(model_path, name)
    sc_ref_f1_score = 0
    for f1 in ref_f1.values():
        sc_ref_f1_score = max(max(f1), sc_ref_f1_score)
    return sc_ref_f1_score

# %% 
sc_ref = load_best_score('ref_f1_score')
print(sc_ref)
mc_ref = load_best_score('multich_ref_f1_score')

sc_wp = load_best_score('wp_f1_score')
print(sc_wp)
mc_wp = load_best_score('multich_wp_f1_score')

sc_temp = load_best_score('spatemp_wp_f1_score')
print(sc_temp)
mc_temp = load_best_score('multich_spatemp_f1_score')

# %%
import matplotlib.pyplot as plt

