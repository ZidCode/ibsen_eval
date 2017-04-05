import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_histogram():
    filenames = ["results/biased_%s_alpha.txt" % i for i in range(30)]

    alpha_matrix = np.array([])
    beta_matrix = np.array([])
    g_dsa_matrix = np.array([])
    g_dsr_matrix = np.array([])
    for file_ in filenames:
        frame = pd.read_csv(file_)
        frame['alpha'] = frame['alpha'] / frame['expected'][0] * 100
        frame['beta'] = frame['beta'] / frame['expected'][1]   * 100
        frame['g_dsa'] = frame['g_dsa'] / frame['expected'][2] * 100
        frame['g_dsr'] = frame['g_dsr'] / frame['expected'][3] * 100

        if file_ == 'results/biased_0_alpha.txt':
             alpha_matrix = frame['alpha']
             beta_matrix =  frame['beta']
             g_dsa_matrix = frame['g_dsa']
             g_dsr_matrix = frame['g_dsr']
        else:
             alpha_matrix =np.vstack(( alpha_matrix,frame['alpha']))
             beta_matrix = np.vstack(( beta_matrix ,frame['beta']))
             g_dsa_matrix =np.vstack(( g_dsa_matrix,frame['g_dsa']))
             g_dsr_matrix =np.vstack(( g_dsr_matrix,frame['g_dsr']))
    plt.errorbar(frame['alpha'], np.mean(beta_matrix, axis=0), yerr=np.std(beta_matrix, axis=0), ecolor ='b' , label=r"$\Delta \beta$", fmt='none')
    plt.errorbar(frame['alpha'], np.mean(g_dsr_matrix,axis=0),yerr=np.std(g_dsr_matrix,axis=0), ecolor= 'g', label=r"$\Delta g_r$", fmt='none')
    plt.errorbar(frame['alpha'], np.mean(g_dsa_matrix, axis=0),yerr=np.std(g_dsa_matrix, axis=0), ecolor= 'r', label=r"$\Delta g_a$", fmt='none')
    plt.legend()
    #plt.title(r"$alpha$:%s,$beta$:%s,$g_{dsa}$:%s,$g_{dsr}$:%s" % (frame['expected'][0], frame['expected'][1], frame['expected'][2], frame['expected'][3]))
    plt.xlabel(r'Angstrom exponent $\Delta \alpha$')
    plt.ylabel(r'$\Delta [\%]$')
    plt.show()

    filenames = ["results/biased_%s_beta.txt" % i for i in range(30)]
    alpha_matrix = np.array([])
    beta_matrix = np.array([])
    g_dsa_matrix = np.array([])
    g_dsr_matrix = np.array([])
    for file_ in filenames:
        frame = pd.read_csv(file_)
        frame['alpha'] = frame['alpha'] / frame['expected'][0] * 100
        frame['beta'] = frame['beta'] / frame['expected'][1]   * 100
        frame['g_dsa'] = frame['g_dsa'] / frame['expected'][2] * 100
        frame['g_dsr'] = frame['g_dsr'] / frame['expected'][3] * 100

        if file_ == 'results/biased_0_beta.txt':
             alpha_matrix = frame['alpha']
             beta_matrix =  frame['beta']
             g_dsa_matrix = frame['g_dsa']
             g_dsr_matrix = frame['g_dsr']
        else:
             alpha_matrix =np.vstack(( alpha_matrix,frame['alpha']))
             beta_matrix = np.vstack(( beta_matrix ,frame['beta']))
             g_dsa_matrix =np.vstack(( g_dsa_matrix,frame['g_dsa']))
             g_dsr_matrix =np.vstack(( g_dsr_matrix,frame['g_dsr']))
    plt.errorbar(frame['beta'], np.mean(alpha_matrix, axis=0), yerr=np.std(alpha_matrix, axis=0), ecolor ='b' , label=r"$\Delta \alpha$", fmt='none')
    plt.errorbar(frame['beta'], np.mean(g_dsr_matrix, axis=0),yerr=np.std(g_dsr_matrix, axis=0), ecolor= 'g', label=r"$\Delta g_r$", fmt='none')
    plt.errorbar(frame['beta'], np.mean(g_dsa_matrix, axis=0),yerr=np.std(g_dsa_matrix, axis=0), ecolor= 'r', label=r"$\Delta g_a$", fmt='none')
    plt.legend()
    #plt.title(r"$alpha$:%s,$beta$:%s,$g_{dsa}$:%s,$g_{dsr}$:%s" % (frame['expected'][0], frame['expected'][1], frame['expected'][2], frame['expected'][3]))
    plt.xlabel(r'Turbidity $\Delta \beta$')
    plt.ylabel(r'$\Delta [\%]$')
    plt.show()


    filenames = ["results/biased_%s_g_dsr.txt" % i for i in range(30)]
    alpha_matrix = np.array([])
    beta_matrix = np.array([])
    g_dsa_matrix = np.array([])
    g_dsr_matrix = np.array([])
    for file_ in filenames:
        frame = pd.read_csv(file_)
        frame['alpha'] = frame['alpha'] / frame['expected'][0] * 100
        frame['beta'] = frame['beta'] / frame['expected'][1]   * 100
        frame['g_dsa'] = frame['g_dsa'] / frame['expected'][2] * 100
        frame['g_dsr'] = frame['g_dsr'] / frame['expected'][3] * 100

        if file_ == 'results/biased_0_g_dsr.txt':
             alpha_matrix = frame['alpha']
             beta_matrix =  frame['beta']
             g_dsa_matrix = frame['g_dsa']
             g_dsr_matrix = frame['g_dsr']
        else:
             alpha_matrix =np.vstack(( alpha_matrix,frame['alpha']))
             beta_matrix = np.vstack(( beta_matrix ,frame['beta']))
             g_dsa_matrix =np.vstack(( g_dsa_matrix,frame['g_dsa']))
             g_dsr_matrix =np.vstack(( g_dsr_matrix,frame['g_dsr']))
    plt.errorbar(frame['g_dsr'], np.mean(alpha_matrix, axis=0), yerr=np.std(alpha_matrix, axis=0), ecolor ='b' , label=r"$\Delta \alpha$", fmt='none')
    plt.errorbar(frame['g_dsr'], np.mean(beta_matrix, axis=0),yerr=np.std(beta_matrix, axis=0), ecolor= 'g', label=r"$\Delta \beta$", fmt='none')
    plt.errorbar(frame['g_dsr'], np.mean(g_dsa_matrix, axis=0),yerr=np.std(g_dsa_matrix, axis=0), ecolor= 'r', label=r"$\Delta g_a$", fmt='none')
    plt.legend()
    #plt.title(r"$alpha$:%s,$beta$:%s,$g_{dsa}$:%s,$g_{dsr}$:%s" % (frame['expected'][0], frame['expected'][1], frame['expected'][2], frame['expected'][3]))
    plt.xlabel(r'Coverty factor $\Delta g_{dsr}$')
    plt.ylabel(r'$\Delta [\%]$')
    plt.show()

    filenames = ["results/biased_%s_g_dsa.txt" % i for i in range(30)]
    alpha_matrix = np.array([])
    beta_matrix = np.array([])
    g_dsa_matrix = np.array([])
    g_dsr_matrix = np.array([])
    for file_ in filenames:
        frame = pd.read_csv(file_)
        frame['alpha'] = frame['alpha'] / frame['expected'][0] * 100
        frame['beta'] = frame['beta'] / frame['expected'][1]   * 100
        frame['g_dsa'] = frame['g_dsa'] / frame['expected'][2] * 100
        frame['g_dsr'] = frame['g_dsr'] / frame['expected'][3] * 100

        if file_ == 'results/biased_0_g_dsa.txt':
             alpha_matrix = frame['alpha']
             beta_matrix =  frame['beta']
             g_dsa_matrix = frame['g_dsa']
             g_dsr_matrix = frame['g_dsr']
        else:
             alpha_matrix =np.vstack(( alpha_matrix,frame['alpha']))
             beta_matrix = np.vstack(( beta_matrix ,frame['beta']))
             g_dsa_matrix =np.vstack(( g_dsa_matrix,frame['g_dsa']))
             g_dsr_matrix =np.vstack(( g_dsr_matrix,frame['g_dsr']))
    plt.errorbar(frame['g_dsa'], np.mean(alpha_matrix, axis=0), yerr=np.std(alpha_matrix, axis=0), ecolor ='b' , label=r"$\Delta \alpha$", fmt='none')
    plt.errorbar(frame['g_dsa'], np.mean(beta_matrix, axis=0),yerr=np.std(beta_matrix,axis=0), ecolor= 'g', label=r"$\Delta \beta$", fmt='none')
    plt.errorbar(frame['g_dsa'], np.mean(g_dsr_matrix, axis=0),yerr=np.std(g_dsr_matrix, axis=0), ecolor= 'r', label=r"$\Delta g_r$", fmt='none')
    plt.legend()
    #plt.title(r"$alpha$:%s,$beta$:%s,$g_{dsa}$:%s,$g_{dsr}$:%s" % (frame['expected'][0], frame['expected'][1], frame['expected'][2], frame['expected'][3]))
    plt.xlabel(r'Coverty factor $\Delta g_{dsa}$')
    plt.ylabel(r'$\Delta [\%]$')
    plt.show()



if __name__ == "__main__":
    plot_histogram()
