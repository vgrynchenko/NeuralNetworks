import numpy as np
import ann as ann
import preprocessing as pp
import os as os
import tqdm as tqdm
import matplotlib.pyplot as plt


def classify(output, marks):
    errors = np.empty(len(marks))
    for i in range(len(marks)):
        errors[i] = np.linalg.norm(output - marks[i])
    return marks[errors.argmin()]


def act(x, a):
    return x


def dact(x, a):
    return a / np.cosh(a*x) / np.cosh(a*x)


if __name__ == '__main__':
    standard_width, standard_height = 60, 60
    path_standard = 'D:/documents/4 course/' \
                    'NN/Standart/'
    path_original = 'D:/documents/4 course/' \
                    'NN/Original/'
    a = 1.0
    expr_num = 20
    w_lower_bound, w_upper_bound = -0.95, 0.95
    class_number = 2
    epoch_n, learning_rate = 1000, 0.01
    folder_of_sample = 'G/', 'V/', 'A/', 'Other/', 'Test/'
    sample, sample_vol = [], 70
    test, test_vol = [], 7
    layer_n, neuron_in_each_n = 5, 2
    neuron_n = np.array([20, 20, 20, 20, class_number])
    marks = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    tmp = [marks[0] for _ in range(test_vol // class_number)]
    tmp.extend([marks[1] for _ in range(test_vol // class_number)])
    test_output = np.array(tmp)
    sample_errors, test_errors = [], []
    next_phase_str = ''.join(['#' for _ in range(20)])
    prefix, postfix = '\n\n', '\n\n'
    figsize = (15.0, 7.5)
    # neuron_n = np.ones(layer_n, dtype=int) * neuron_in_each_n
    # neuron_n = np.arange(6, 3, -1)
    print(prefix + next_phase_str + '\nПредобработка исходной выборки\n' + next_phase_str + postfix)
    pp.trim_edges(path_original, path_standard, folder_of_sample, folder_of_sample, True)
    pp.standardize_size(path_standard, path_standard, folder_of_sample, folder_of_sample,
                        standard_width, standard_height, True)
    print(prefix + next_phase_str + '\nСоставление выборки\n' + next_phase_str + postfix)
    for cl in range(class_number):
        examples_names = [file for file in os.listdir(path_standard + folder_of_sample[cl]) if file.endswith('.jpg')]
        for ex in tqdm.tqdm(examples_names):
            sample.append(pp.image_to_matrix(path_standard + folder_of_sample[cl] + ex).ravel())
    sample = np.array(sample)
    test_names = [file for file in os.listdir(path_standard + folder_of_sample[-1]) if file.endswith('.jpg')]
    for t in tqdm.tqdm(test_names):
        test.append(pp.image_to_matrix(path_standard + folder_of_sample[-1] + t).ravel())
    test = np.array(test)
    output = np.vstack((marks[i] * np.ones((sample_vol, class_number)) for i in range(class_number)))
    pl = ann.PerceptronLgst(sample.shape[1], layer_n, neuron_n, learning_rate=learning_rate, epoch_n=epoch_n,
                            activation_param=a, w_lower_bound=w_lower_bound, w_upper_bound=w_upper_bound)
    print(prefix + next_phase_str + '\nВывод нейронной сети с неоптимизированными весами\n' + next_phase_str + postfix)
    for i in range(sample.shape[0]):
        print('{0} - {1}'.format(i, classify(pl.predict(sample[i]), marks)))
    print(prefix + next_phase_str + '\nОптимизация весов\n' + next_phase_str + postfix)
    pl.fit(sample, output)
    print(prefix + next_phase_str + '\nОшибки обучения\n' + next_phase_str + postfix)
    print(pl.errors)
    print(prefix + next_phase_str + '\nВывод нейронной сети с оптимизированными весами\n' + next_phase_str + postfix)
    for i in range(sample.shape[0]):
        print('{0} - {1}'.format(i, classify(pl.predict(sample[i]), marks)))
    print(prefix + next_phase_str + '\nОбобщение нейронной сети на тестовой выборке\n' + next_phase_str + postfix)
    for i in range(test.shape[0]):
        print('{0} - {1}'.format(test_names[i], classify(pl.predict(test[i]), marks)))
    plt.figure(figsize=figsize)
    plt.plot(np.arange(1, epoch_n + 1), pl.errors, 'r-')
    plt.grid(True)
    plt.show()
    plt.close()