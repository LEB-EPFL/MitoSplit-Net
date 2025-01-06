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
import json
from datetime import datetime

sc_ref = load_best_score('ref_f1_score')
mc_ref = load_best_score('multich_ref_f1_score')
sc_wp = load_best_score('wp_f1_score')
mc_wp = load_best_score('multich_wp_f1_score')
sc_temp = load_best_score('spatemp_wp_f1_score')
mc_temp = load_best_score('multich_spatemp_f1_score')

# Organize the data
data = {
    "timestamp": datetime.now().isoformat(),
    "scores": {
        "sc_ref": sc_ref,
        "mc_ref": mc_ref,
        "sc_wp": sc_wp,
        "mc_wp": mc_wp,
        "sc_temp": sc_temp,
        "mc_temp": mc_temp
    }
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"scores_data_{timestamp}.json"

#%%
# Save the data to a timestamped JSON file
with open('data/' + file_name, "w") as f:
    json.dump(data, f, indent=4)

print(f"Data saved to {file_name}")


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

data_files = Path('data').glob('scores_data*.json')
categories = ['ref', 'wp', 'temp']
n_channels = ['sc', 'mc']
long_data = []
for data_file in data_files:
    data = json.load(open(data_file))
    for cat in categories:
        for n in n_channels:
            long_data.append({"Category": cat, "Score": data['scores'][f"{n}_{cat}"], "Channel": n})
    
df = pd.DataFrame(long_data)
fig= plt.figure(figsize=(10, 6))
sns.boxplot(x="Category", y="Score", hue="Channel", data=df, palette="Set2")
sns.swarmplot(x="Category", y="Score", hue="Channel", data=df, color=".25", dodge=True)

plt.xlabel("Ground Truth Optimizations")
plt.ylabel("F1 Score")
plt.title("Model Performance Trained With Optimized Empirical Ground Truth")
# plt.xticks(rotation=45)
plt.gca().set_xticklabels(['Reference', 'Spatial', 'Temporal'])
plt.tight_layout()
plt.show()


# %%
