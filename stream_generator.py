import numpy as np
from scipy.ndimage import gaussian_filter



def hyper_abrupt(systhetic_param):
    stream_size = systhetic_param['stream_size']
    noise_var = systhetic_param['noise_var']
    hyperplane_dimension = systhetic_param['dim']
    drift_probability = systhetic_param['drift_prob']

    stream = []
    # initialize and normalize w
    w = np.random.normal(0,scale=1,size=hyperplane_dimension)
    w = w / np.linalg.norm(w)
    for k in range(stream_size):
        if np.random.uniform() < drift_probability and drift_probability != -1:
            w = np.random.normal(0, scale=1, size=hyperplane_dimension)
            w = w / np.linalg.norm(w)
        if drift_probability == -1 and k == int(stream_size/2):
            w = w / np.linalg.norm(w)
            w = np.random.normal(0, scale=1, size=hyperplane_dimension)
        # draw uniform random samples 
        X = np.random.uniform(-10, 10, hyperplane_dimension)
        # create the target parameter using the features
        y = np.dot(X,w)
        # make the stream noisy
        y = y + np.random.normal(0, noise_var)
        stream.append([X, y, w])
    return stream


def hyper_gaussian(systhetic_param, smoothness=50):
    stream_size = systhetic_param['stream_size']
    noise_var = systhetic_param['noise_var']
    hyperplane_dimension = systhetic_param['dim']

    stream = []
    # initialize and normalize w_list
    w_list = np.zeros([stream_size, hyperplane_dimension])
    for d in range(hyperplane_dimension):
        white_noise = np.random.normal(0,10,stream_size)
        w_list[:,d] = gaussian_filter(white_noise, sigma=smoothness)
    for k in range(stream_size):
        w_list[k,:] = w_list[k,:] / np.linalg.norm(w_list[k,:])
    for k in range(stream_size):
        w = w_list[k,:]
        # draw uniform random samples 
        X = np.random.uniform(-10, 10, hyperplane_dimension)
        # create the target parameter using the features
        y = np.dot(X,w)
        # make the stream noisy
        y = y + np.random.normal(0, noise_var)
        stream.append([X, y, w])
    return stream


def hyper_random_walk(systhetic_param, random_walk_noise=0.01):
    stream_size = systhetic_param['stream_size']
    noise_var = systhetic_param['noise_var']
    hyperplane_dimension = systhetic_param['dim']
    
    stream = []
    # initialize and normalize w
    w = np.random.normal(0,scale=1,size=hyperplane_dimension)
    w = w / np.linalg.norm(w)
    for k in range(stream_size):
        w = w + np.random.normal(0,scale=random_walk_noise, size=hyperplane_dimension)
        w = w / np.linalg.norm(w)
        # draw uniform random samples 
        X = np.random.uniform(-10, 10, hyperplane_dimension)
        # create the target parameter using the features
        y = np.dot(X,w)
        # make the stream noisy
        y = y + np.random.normal(0, noise_var)
        stream.append([X, y, w])
    return stream

def hyper_gradual(systhetic_param, drift_duration_min=10, drift_duration_max=200):
    stream_size = systhetic_param['stream_size']
    noise_var = systhetic_param['noise_var']
    hyperplane_dimension = systhetic_param['dim']
    drift_probability = systhetic_param['drift_prob']
    stream = []
    # initialize and normalize w
    w = np.random.normal(0,scale=1,size=hyperplane_dimension)
    w = w / np.linalg.norm(w)
    s = None
    for k in range(stream_size):
        if s is None and np.random.rand() < drift_probability:
                s = 0
                w_old = w
                w_new = np.random.normal(0,scale=1,size=hyperplane_dimension)
                w_new = w_new / np.linalg.norm(w_new)
                drift_duration = np.random.uniform(drift_duration_min, drift_duration_max)
        elif s is not None:
            if 0 <= s < 1:
                if np.random.rand() > s:
                    w = w_old
                else:
                    w = w_new
                s = s + 1/drift_duration
            elif s >= 1:
                w = w_new
                s = None
        # draw uniform random samples 
        X = np.random.uniform(-10, 10, hyperplane_dimension)
        # create the target parameter using the features
        y = np.dot(X,w)
        # make the stream noisy
        y = y + np.random.normal(0, noise_var)
        stream.append([X, y, w])
    return stream


def hyper_incremental(systhetic_param, drift_duration_min=10, drift_duration_max=200):
    stream_size = systhetic_param['stream_size']
    noise_var = systhetic_param['noise_var']
    hyperplane_dimension = systhetic_param['dim']
    drift_probability = systhetic_param['drift_prob']
    stream = []
    # initialize and normalize w
    w = np.random.normal(0,scale=1,size=hyperplane_dimension)
    w = w / np.linalg.norm(w)
    s = None
    for k in range(stream_size):
        if s is None and np.random.uniform() < drift_probability:
                s = 0
                w_old = w
                w_new = np.random.normal(0,scale=1,size=hyperplane_dimension)
                w_new = w_new / np.linalg.norm(w_new)
                drift_duration = np.random.uniform(drift_duration_min, drift_duration_max)
        elif s is not None:
            if 0 <= s < 1:
                w = (1-s)*w_old + s*w_new
                w = w / np.linalg.norm(w)
                s = s + 1/drift_duration
            elif s >= 1:
                w = w_new
                s = None
        # draw uniform random samples 
        X = np.random.uniform(-10, 10, hyperplane_dimension)
        # create the target parameter using the features
        y = np.dot(X,w)
        # make the stream noisy
        y = y + np.random.normal(0, noise_var)
        stream.append([X, y, w])
    return stream


def hyper_linear(synthetic_param):
    stream_size = synthetic_param['stream_size']
    noise_var = synthetic_param['noise_var']
    hyperplane_dimension = synthetic_param['dim']
    stream = []
    # initialize and normalize w
    w = np.random.normal(0,scale=1,size=hyperplane_dimension)
    w = w / np.linalg.norm(w)
    for k in range(stream_size):
        if k == 0:
                random_direction = np.random.normal(0,scale=0.01, size=hyperplane_dimension)
                random_direction = random_direction / np.linalg.norm(random_direction)
        else:
            w = w + (1/stream_size)*(random_direction - w)
            w = w / np.linalg.norm(w)
        # draw uniform random samples 
        X = np.random.uniform(-10, 10, hyperplane_dimension)
        # create the target parameter using the features
        y = np.dot(X,w)
        # make the stream noisy
        y = y + np.random.normal(0, noise_var)
        stream.append([X, y, w])
    return stream


def simple_heterogeneous(synthetic_param):
    stream_size = synthetic_param['stream_size']
    noise_var = synthetic_param['noise_var']
    # m = np.random.uniform(-10, 10, 1)
    m = 1
    stream = []
    for k in range(stream_size):
        # draw uniform random samples 
        if k > int(stream_size/2):
            # m = np.random.uniform(-10, 10, 1)
            m = -1
        X = np.random.uniform(-10, 10, 1)
        # create the target parameter using the features
        if X < 0:
            y = 0
        else:
            y = float(m * X[0])
        # make the stream noisy
        y = np.random.normal(y, noise_var)
        # print(X,y,'\n')
        stream.append([X, y])
    return stream


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # hyper_gaussian(stream_size=1000, noise_var=0.05, hyperplane_dimension=5, smoothness = 50)
    # hyper_gradual(stream_size=1000, noise_var=0.05, hyperplane_dimension=5, drift_duration_min=10, drift_duration_max=200, drift_duration=200, drift_probability=0.05)
    # hyper_incremental(stream_size=1000, noise_var=0.05, hyperplane_dimension=5, drift_duration_min=10, drift_duration_max=200, drift_probability=0.01)
    # hyper_linear(stream_size=1000, noise_var=0.05, hyperplane_dimension=5)
    stream = hyper_abrupt(systhetic_param={'stream_size':10000, 'noise_var':0.05, 'dim':5, 'drift_prob':0.05})
    # stream = hyper_random_walk(stream_size=10000, noise_var=0.05, hyperplane_dimension=5, random_walk_noise=0.01)

    # np.savetxt("hyperplane_data.txt", np.array([item[0:2] for item in stream]))

    with open('hyperplane_data.txt', 'w') as file:
        for i in range(len(stream[0][0])):
            if i == 0:
                file.write('feature_'+str(i+1))
            else:
                file.write(', '+'feature_'+str(i+1))
        file.write(', '+'target_feature\n')
        for item in stream:
            file.write(','.join(map(str, item[0])))
            file.write(', '+ str(float(item[1])))
            file.write('\n')

    # plot hyperplane parameters (normal vector elements)
    plt.plot([item[2] for item in stream])
    plt.xlabel('Stream timestep')
    plt.ylabel('Hyperplane parameters')
    plt.show()


