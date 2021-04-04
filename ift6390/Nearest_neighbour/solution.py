import numpy as np
# import matplotlib.pyplot as plt

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:
    # def __init__(self):
    #     self.data = np.genfromtxt('data_banknote_authentication.txt', delimiter = ',')

    def feature_means(self, banknote):
        return np.mean(banknote[:,:4], axis=0)

    def covariance_matrix(self, banknote):
        return np.cov(banknote[:,:4].T, bias=True)

    def feature_means_class_1(self, banknote):
        arr = banknote[banknote[:,4]==1]
        return self.feature_means(arr)

    def covariance_matrix_class_1(self, banknote):
        arr = banknote[banknote[:,4] == 1]
        return self.covariance_matrix(arr)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def minkowski_mat(self, x, Y, p=2):
        return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels.astype('int')
        self.n_classes = len(np.unique(self.train_labels))
        self.labels = np.unique(self.train_labels)

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]
        # counts = np.ones((num_test, self.n_classes))
        classes_pred = np.zeros(num_test)

        for (i, ex) in enumerate(test_data):

            dist = self.minkowski_mat(self.train_inputs, ex)
            cls = np.zeros(self.n_classes)

            val = self.train_labels[dist<=self.h]
            for idx in range(len(val)):
                cls[val[idx]] += 1



            if len(val):
                classes_pred[i] = np.argmax(cls)
            else:
                # count = 
                classes_pred[i] = draw_rand_label(ex, self.labels)

        return classes_pred.astype('int')


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma = sigma

    def minkowski_mat(self, x, Y, p=2):
        # print(np.sum(np.abs(x-Y)**p,))
        # print(Y.shape)
        return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels.astype('int')
        self.n_classes = len(np.unique(self.train_labels))
        self.labels = np.unique(self.train_labels)
        self.d = len(train_inputs[0])

    def parzen_window(self, ex, d, train):
        part1 = 1 / (((2 * np.pi) ** (d / 2)) * ((self.sigma) ** d))
        part2 = (-1 / 2) * ((self.minkowski_mat(train, ex) / self.sigma) ** 2)
        # print(part1)
        # print(part2)
        final = part1 * np.exp(part2)
        # print(final.shape)
        # break
        return final

    def compute_predictions(self, test_data):
        classes = np.zeros(len(test_data))
        for (i, ex) in enumerate(test_data):
            cls = np.zeros(self.n_classes)
            # print(cls)
            # print('cls')
            label = self.train_labels.astype('int')
            train = self.train_inputs
            
            prob = self.parzen_window(ex, self.d, train)
            for (j, sample) in enumerate(prob):
              cls[label[j]] += sample

            classes[i] = np.argmax(cls)

        return classes.astype('int')

def split_dataset(banknote):
    train_ind, test_ind, val_ind = [], [], []
    for i in range(len(banknote)):
        if i % 5 == 3:
            val_ind.append(i)
        elif i % 5 == 4:
            test_ind.append(i)
        else:
            train_ind.append(i)
    train = banknote[train_ind]
    val = banknote[val_ind]
    test = banknote[test_ind]
    return (train, val, test)



class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        Hp = HardParzen(h)
        Hp.train(self.x_train, self.y_train)
        pred = Hp.compute_predictions(self.x_val)
        total_num = len(self.y_val)
        num_correct = np.sum(pred == self.y_val)
        error_rate = 1.0 - (float(num_correct) / float(total_num))
        return error_rate


    def soft_parzen(self, sigma):
        Sp = SoftRBFParzen(sigma)
        Sp.train(self.x_train, self.y_train)
        pred = Sp.compute_predictions(self.x_val)
        total_num = len(self.y_val)
        num_correct = np.sum(pred == self.y_val)
        error_rate =  1.0 - (float(num_correct) / float(total_num))
        return error_rate
'''
def plot(banknote):
    h = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    train, val, test = split_dataset(banknote)
    x_train = train[:,:-1]
    y_train = train[:,-1].astype('int')
    x_val = val[:,:-1]
    y_val = val[:,-1].astype('int')
    test_error = ErrorRate(x_train, y_train, x_val, y_val)
    hp_error = []
    sp_error = []
    for i in h:
        a = test_error.hard_parzen(i)
        b = test_error.soft_parzen(i)
        hp_error.append(a)
        sp_error.append(b)
    plt.plot(h, hp_error, label='hardparzen error')
    plt.plot(h, sp_error, label='softparzen error')
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('h/sigma value')
    plt.show()
    plt.savefig('Question_5.png')
'''
def get_test_errors(banknote):
    h = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]

    train, val, test = split_dataset(banknote)
    x_train = train[:,:-1]
    y_train = train[:,-1].astype('int')
    x_val = val[:,:-1]
    y_val = val[:,-1].astype('int')
    x_test = test[:,:-1]
    y_test = test[:,-1].astype('int')
    val_error = ErrorRate(x_train, y_train, x_val, y_val)
    hp_error = []
    sp_error = []
    for i in h:
        a = val_error.hard_parzen(i)
        b = val_error.soft_parzen(i)
        hp_error.append(a)
        sp_error.append(b)

    h_min = h[np.argmin(hp_error)]
    s_min = h[np.argmin(sp_error)]
    # print('H_min : {} , S_min: {}'.format(h_min, s_min))

    test_error = ErrorRate(x_train, y_train, x_test, y_test)
    a = test_error.hard_parzen(h_min)
    b = test_error.soft_parzen(s_min)
    return np.array([a, b])



def random_projections(X, A):
    return (1/np.sqrt(2)) * np.dot(X,A)

'''
def random_projection_n(banknote, n= 10):
    (train, val, test) = split_dataset(banknote)
    train_labels = train[:,4:5]
    val_labels = val[:,4:5]
    train_inputs = train[:,0:4]
    val_inputs = val[:,0:4]

    h = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    error_matrix_hard = np.zeros([n,len(h)])
    error_matrix_soft = np.zeros([n,len(h)])

    for i in range(n):
        A = np.random.normal(0,1, (4,8))
        train_inputs = random_projections(train[:,0:4], A)
        val_inputs = random_projections(val[:,0:4], A)
        err = ErrorRate(train_inputs, train_labels, val_inputs, val_labels)
        for j,value in enumerate(h):
            error_matrix_hard[i][j] = err.hard_parzen(value)
            error_matrix_soft[i][j] = err.soft_parzen(value)
        print('---------------')
        print(error_matrix_hard[i])
        print(error_matrix_soft[i])

    hard = np.mean(error_matrix_hard, axis=0)
    soft = np.mean(error_matrix_soft, axis=0)

    plt.figure()
    plt.errorbar(h, hard, yerr=0.2)
    plt.errorbar(h, soft, yerr=0.2)
    plt.show()
    plt.savefig('Question_9.png')
    # return error_matrix_hard, error_matrix_soft

'''
