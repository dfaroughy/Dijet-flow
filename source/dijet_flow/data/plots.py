import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dijet_flow.utils.base import savefig
from dijet_flow.utils.collider import deltaR, dphi, deta

sns.set_theme(style="dark")

def plot_loss(train, valid, args):
    train_loss = train.loss_per_epoch
    valid_loss = valid.loss_per_epoch
    loss_min = valid.loss_min
    fig, ax = plt.subplots(figsize=(8,7))
    plt.plot(range(len(train_loss)), np.array(train_loss), color='b', lw=0.75)
    plt.plot(range(len(valid_loss)), np.array(valid_loss), color='r', lw=0.75, alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("loss_min={}, epochs={}".format(round(loss_min,6),len(train_loss)))
    fig.tight_layout()
    plt.grid() 
    plt.savefig(savefig(args.workdir+'/loss.png', extension="png"))


def jet_plot_routine( data, 
                      title, 
                      save_dir,  
                      bins=100,
                      figsize=(20, 15)
                    ):

    print("INFO: plotting -> {}".format(title))
    jet1 = [ data[:, 0], data[:, 1], data[:, 2], data[:, 3] ]
    jet2 = [ data[:, 4], data[:, 5], data[:, 6], data[:, 7] ]
    low_level_feat = [r'$p_t$ (GeV)',r'$\eta$',r'$\phi$',r'$m$ (GeV)']
    xlim_llf = [(500, 2500), (-.005, .005), (-.005, .005), (0, 800)]

    mjj = data[:,-1]
    del_phi = dphi(data[:, :4], data[:, 4:8])
    del_eta = deta(data[:, :4], data[:, 4:8])
    del_R = deltaR(data[:, :4], data[:, 4:8])
    dijet = [mjj, del_eta, del_phi, del_R ]
    high_level_feat = [r'$m_{jj}$ (GeV)', r'$\Delta\eta$',r'$\Delta\phi$',r'$\Delta R$']
    xlim_hlf = [(3100, 3900), (-.005, .005), (-.005, .005), (0, .005)]


    fig, axes = plt.subplots(3, 4, figsize=figsize)

    for idx, llf  in enumerate(low_level_feat):

        sns.histplot(x=jet1[idx], bins=bins, color='k', ax=axes[0, idx], element="step", kde = True, alpha=0.4)
        axes[0, idx].set_xlim(xlim_llf[idx])
        axes[0, idx].set_xlabel(llf+' jet 1')
        axes[0, idx].set_ylabel('counts')
        axes[0, idx].grid()

        sns.histplot(x=jet2[idx], bins=bins, color='k', ax=axes[1, idx], element="step", kde = True, alpha=0.4)
        axes[1, idx].set_xlim(xlim_llf[idx])
        axes[1, idx].set_xlabel(llf+' jet 2')
        axes[1, idx].set_ylabel('counts')
        axes[1, idx].grid()
    
    for idx, hlf  in enumerate(high_level_feat):
        sns.histplot(x=dijet[idx], bins=bins, color='k', ax=axes[2, idx], element="step", kde = True, alpha=0.4)
        axes[2, idx].set_xlim(xlim_hlf[idx])
        axes[2, idx].set_xlabel(hlf)
        axes[2, idx].set_ylabel('counts')
        axes[2, idx].grid()

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(save_dir+'/{}.png'.format(title.replace(" ", "_")))

