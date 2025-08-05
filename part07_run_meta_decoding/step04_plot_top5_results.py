# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
fig, axs = plt.subplots(3, 1, figsize = (7, 8) )

for i_idx, i_p in enumerate(['p2p1', 'p3p1', 'p4p1']):
    with open(f"./{i_p}_corrdecoder.json", mode = "r", encoding = "utf-8") as f:
        decoded_rdict = json.load(f)
        
    plt_dat = dict({
    'label': [],
    'rhor': [],
    'pval': [],
    'conflw': [],
    'confup': [],
    })
    
    th_top5 = np.sort(np.array([
        _['rhor'] for _ in decoded_rdict.values() 
        if _['newlab'][-1] != ')']))[-5]
    for dec_k, dec_v in decoded_rdict.items():
        if (dec_v['rhor'] >= th_top5) & (dec_v['newlab'][-1] != ')'):
            #plt_dat['label'].append(re.sub(r'LDA50_abstract_weight__', 'Topic', dec_k))
            plt_dat['label'].append(dec_v['newlab'][9:])
            plt_dat['rhor'].append(dec_v['rhor'])
            plt_dat['pval'].append(dec_v['pval'])
            plt_dat['conflw'].append(dec_v['conflw'])
            plt_dat['confup'].append(dec_v['confup'])
    
    plt_dat = pd.DataFrame(plt_dat).sort_values('rhor', ascending=False).reset_index(drop = True)
    
    sns.barplot(plt_dat, x= "rhor", y = "label", ax = axs[i_idx])
    axs[i_idx].errorbar(
        x = plt_dat['rhor'],
        y = plt_dat['label'],
        xerr = np.array([
            list(plt_dat['rhor'] - plt_dat['conflw']), 
            list(plt_dat['confup'] - plt_dat['rhor'])]),
        fmt = 'none',
        ecolor = 'black')
    axs[i_idx].set_title(f"{i_p}")
    axs[i_idx].set_yticklabels(plt_dat['label'])
    axs[i_idx].set_xlim(0, 0.29)
plt.tight_layout()
fig.savefig("./funcdecoding_p2p1_p3p1_p4p1.pdf")