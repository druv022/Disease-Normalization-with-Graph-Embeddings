import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

def plot_heatmap(tokens, matrix, file_path='None'):

    plt.figure(figsize=(15,10))
    sns.heatmap(matrix, cmap='Blues', xticklabels=tokens, yticklabels=tokens)
    # plt.show()
    plt.savefig(file_path)
    plt.close()


# if __name__ == '__main__':
#     x = np.random.rand(10,10)

#     y = ['asdfghjklasdfghjkl','1','2','3','4','5','6','7','8','9']

#     # sns.heatmap(x,cmap='GnBu', xticklabels=y, yticklabels=y)

#     # plt.show()

#     plot_heatmap(y, x)

#     print('Here')
    



