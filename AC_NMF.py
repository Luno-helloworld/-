import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('C:/Users/17839/Downloads/letter118.crop.wav')
S = np.abs(librosa.stft(y))
# add all of the observations in Nodes
node = [S for _ in range(5)]
Nodes = np.stack(node,axis = 0)
num_Nodes = Nodes.shape[0]

def ac_nmf(X, l, maxiter,  rho): #调用此函数时的输入X应为Nodes
    W = []
    H = []
    U_W = []
    U_H = []

    row, col = Nodes[0].shape

    W_hat = np.ones((row, l))
    H_hat = np.ones((l, col))
    U_W_hat = np.ones((row, l))
    U_H_hat = np.ones((l, col))

    # the global variables
    Z_w = np.ones((row, l))
    Z_h = np.ones((l, col))
    obj = []
    iter = 0
    while iter < maxiter:

        # initialization
        for i in range(num_Nodes):

            X_i = Nodes[i]
            np.random.seed(123)
            # 0-1
            W_i = np.array(np.random.rand(row, l))
            H_i = np.array(np.random.rand(l, col))
            # init the dual variables
            alpha_W_i = np.zeros_like(W_i)
            alpha_H_i = np.zeros_like(H_i)
            U_H_i = 1 / rho * alpha_H_i
            U_W_i = 1 / rho * alpha_W_i

            d_W1_i = np.dot(H_i, H_i.T) + rho * np.eye(l)
            d_W2_i = np.linalg.inv(d_W1_i)
            m_W_i = np.dot(X_i, H_i.T) + rho * (Z_w - U_W_i)
            W_i = np.dot(m_W_i, d_W2_i)

            d_H1_i = np.dot(W_i.T, W_i) + rho * np.eye(l)
            d_H2_i = np.linalg.inv(d_H1_i)
            m_H_i = np.dot(W_i.T, X_i) + rho * (Z_h - U_H_i)
            H_i = np.dot(d_H2_i, m_H_i)

            Z_w = np.maximum(W_hat + U_W_hat, 0)
            Z_h = np.maximum(H_hat + U_H_hat, 0)

            U_W_i = U_W_i + (W_i - Z_w)
            U_H_i = U_H_i + (H_i - Z_h)

            W.append(W_i)
            H.append(H_i)
            U_W.append(U_W_i)
            U_H.append(U_H_i)

            W_hat = np.mean(W)
            H_hat = np.mean(H)
            U_W_hat = np.mean(U_W)
            U_H_hat = np.mean(U_H)

            err = [] # the error of every iterration
            err_i = np.linalg.norm(X_i - np.dot(W_i, H_i))
            err.append(err_i)
            err_sum = sum(err) # the objective function in each iteration



        iter += 1
        obj.append(err_sum)

    return W, H, obj

if __name__ == "__main__":
    X = Nodes
    W, H, obj = ac_nmf(X, 5, 10,0.01)
    x = range(len(obj))

    plt.plot(x, obj)


    plt.show()









