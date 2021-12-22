from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N_ROTATIONS = 4


def get_new_etas_estimate(etas_estimate: Tuple[float,float], fixed_shape: np.array, avg_image: np.array) \
        -> Tuple[Tuple[float,float], bool]:
    np.random.seed(1)
    def func(etas):
        eta1, eta2 = etas
        eta_matrix = np.where(fixed_shape == 1, eta1, eta2)
        dotproduct = eta_matrix * avg_image - np.log(1+np.exp(eta_matrix))
        result_for_background = np.sum(dotproduct[np.where(avg_image>0.5)])
        result_for_foreground = np.sum(dotproduct[np.where(avg_image<=0.5)])
        result = result_for_foreground + result_for_background
        return -result
    estimate = list(etas_estimate)
    res = minimize(func, estimate, method="Nelder-Mead")
    new_eta1, new_eta2 = res.x
    if new_eta1 != etas_estimate[0] or new_eta2 != etas_estimate[1]:
        etas_changed = True
    else:
        etas_changed = False
    return (new_eta1, new_eta2), etas_changed



def get_new_shape_estimate(fixed_etas: Tuple[float,float], shape_estimate: np.array, avg_image: np.array) \
        -> Tuple[np.array, bool]:
    eta1, eta2 = fixed_etas
    matrix_eta1 = avg_image * eta1 - np.log(1 + np.exp(eta1))
    matrix_eta2 = avg_image * eta2 - np.log(1 + np.exp(eta2))
    new_shape = matrix_eta2 < matrix_eta1
    shape_changed = False in (new_shape == shape_estimate)
    return new_shape, shape_changed


def shape_mle(avg_image: np.array, etas_init: Tuple[float, float]) -> tuple[np.array, tuple[float, float]]:
    shape_changed = True
    etas_changed = True
    etas_estimate = etas_init
    shape_estimate = np.random.normal(size=avg_image.shape) > 0
    shape_initial = shape_estimate
    counter = 0
    while shape_changed or etas_changed:
        if counter % 10 == 0: print("Running iteration: ",counter)
        etas_estimate, etas_changed = get_new_etas_estimate(etas_estimate, shape_estimate, avg_image)
        shape_estimate, shape_changed = get_new_shape_estimate(etas_estimate, shape_estimate, avg_image)
        counter += 1
        # plt.imshow(shape_estimate, cmap='Greys', interpolation='nearest')
        # plt.show()
    print("[shape_mle()]: Optimization completed..")
    plt.subplot(211)
    plt.title("Initial shape estimate")
    plt.imshow(shape_initial, cmap='Greys',  interpolation='nearest')
    plt.subplot(212)
    plt.title("Shape estimate after {} optimization iterations".format(counter+1))
    plt.imshow(shape_estimate, cmap='Greys', interpolation='nearest')
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
    plt.savefig("shape_mle_result_first_task.png")
    return shape_estimate,(etas_estimate[0], etas_estimate[1])

def posterior_pose_probs(images: np.array, shape_estimate: np.array, etas_estimate: np.array, poses_estimate: np.array) -> np.array:
    eta1,eta0 = etas_estimate
    joined_images = np.stack([images] * N_ROTATIONS, axis=0)
    shape90 = np.rot90(shape_estimate, axes=(0, 1),k=1)
    shape180 = np.rot90(shape_estimate, axes=(0, 1),k=2)
    shape270 = np.rot90(shape_estimate, axes=(0, 1),k=3)
    joined_shapes = np.stack([shape_estimate,shape90,shape180, shape270], axis=0)
    joined_shapes = np.broadcast_to(joined_shapes[:,np.newaxis,:,:],
                                    (4, joined_images.shape[1],
                                     joined_images.shape[2],
                                     joined_images.shape[3]))
    joined_shapes = np.where(joined_shapes == 1, eta1, eta0)
    poses_estimate = np.broadcast_to(poses_estimate, (joined_images.shape[1], 4))
    dotproduct = np.sum(joined_shapes*joined_images, axis = (3,2)).T
    softmaxinput = dotproduct + np.log(poses_estimate)
    alphas = np.divide(np.exp(softmaxinput), np.reshape(np.sum(np.exp(softmaxinput), axis=1), (softmaxinput.shape[0], 1)))
    return alphas

def get_average_image(data: np.array, alphas: np.array) -> np.array:
    data90 = np.rot90(data, axes=(1, 2), k=N_ROTATIONS-1)
    data180 = np.rot90(data, axes=(1, 2), k=N_ROTATIONS-2)
    data270 = np.rot90(data, axes=(1, 2), k=N_ROTATIONS-3)
    joined_images = np.stack([data,data90,data180, data270], axis=0)
    alphas = np.broadcast_to(alphas.T[:,:,np.newaxis,np.newaxis],(N_ROTATIONS,data.shape[0],data.shape[1],data.shape[2]))
    avg_image = np.sum(alphas*joined_images, axis = (0,1)) / data.shape[0]
    return avg_image


def get_new_poses_estimate(alphas: np.array, pose_probs: np.array) -> np.array:
    def func(pose_probs):
        result = alphas * np.log(np.array(pose_probs))
        result = np.sum(result, axis = (0,1))
        return - result
    def constraint1(x):
        return sum(x)-1
    estimate = list(pose_probs)
    con1 = {'type': 'eq', 'fun': constraint1}
    res = minimize(func, estimate, bounds=((0,1),(0,1),(0,1),(0,1)), options= {"maxiter":100000000},constraints=con1)
    return res.x


def em_algorithm(images: np.array, etas_init: Tuple[float, float], shape_estimate: np.array, pose_estimate: np.array) \
    -> Tuple[np.array, Tuple[float,float], np.array]:
    em_steps = 1
    def e_step(images: np.array, shape_estimate: np.array, etas_estimate: np.array, poses_estimate: np.array):
        return posterior_pose_probs(images, shape_estimate, etas_estimate, poses_estimate)
    def m_step(alphas: np.array, average_image: np.array, etas_estimate: np.array, shape_estimate: np.array, poses_estimate: np.array):
        shape_changed = True
        etas_changed = True
        poses_estimate = get_new_poses_estimate(alphas, poses_estimate)
        while shape_changed or etas_changed:
            etas_estimate, etas_changed = get_new_etas_estimate(etas_estimate, shape_estimate, average_image)
            shape_estimate, shape_changed = get_new_shape_estimate(etas_estimate, shape_estimate, average_image)
        print("New estimates in M step {}: \n\tetas {}, \n\tpose probabilities {}".format(em_steps, etas_estimate, [np.round(x,4) for x in poses_estimate]))
        print("----------------------------")
        print()
        return etas_estimate, shape_estimate, poses_estimate

    shape_initial = shape_estimate
    alphas = np.random.normal(size=(images.shape[0], N_ROTATIONS)) > 0
    etas_estimate = etas_init
    while True:
        print("Running EM step {}..".format(em_steps))
        new_alphas = e_step(images,shape_estimate, etas_estimate, pose_estimate)
        if (new_alphas == alphas).all():
            print("EM termination condition reached")
            break
        else:
            alphas = new_alphas
        etas_estimate, shape_estimate, pose_estimate = m_step(alphas,get_average_image(images,alphas),etas_estimate,shape_estimate,pose_estimate)
        em_steps += 1
    print("[em_algorithm()]: Optimization completed in {} EM iterations..".format(em_steps))
    plt.subplot(211)
    plt.title("Initial shape estimate")
    plt.imshow(shape_initial, cmap='Greys', interpolation='nearest')
    plt.subplot(212)
    plt.title("Shape estimate after {} EM iterations".format(em_steps + 1))
    plt.imshow(shape_estimate, cmap='Greys', interpolation='nearest')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig("shape_mle_result_second_task.png")
    return shape_estimate,etas_estimate, pose_estimate

if __name__ == '__main__':

    print("-- Assignment 1 --")
    data = np.load("em_data/images0.npy")
    meanimage = np.mean(data, axis=0)
    eta1_initial = np.log(0.7 / (1 - 0.7))
    eta0_initial = np.log(0.3 / (1 - 0.3))
    _, (eta1, eta0) = shape_mle(meanimage,(eta1_initial,eta0_initial))
    print("Eta1 is {} and Eta0 is {}".format(eta1,eta0))
    print("-- End of Assignment 1 --")



    print("-- Assignment 2 --")
    data = np.load("em_data/images.npy")
    meanimage = np.mean(data,axis=0)
    eta1_initial = np.log(0.7/(1-0.7))
    eta0_initial = np.log(0.3/(1-0.3))
    shape_estimate = np.random.normal(size=data[0].shape) > 0
    pose_estimate = np.array([0.25,0.25,0.25,0.25])
    _, (eta1, eta0), pose_estimate = em_algorithm(data, (eta1_initial, eta0_initial), shape_estimate, pose_estimate)
    print("Eta1 is {} and Eta0 is {}".format(eta1,eta0))
    print("Pose estimates are {}".format(pose_estimate))
    print("-- End of Assignment 2 --")
