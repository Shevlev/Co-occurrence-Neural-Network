
'''
Create the toy dataset
'''
import matplotlib.pyplot as plt
import numpy as np

SIZE_IMAGE = 10
NUM_BINS = 4
HIGH_DIST = 0.4
LOW_DIST = 0.1
# number_images per class
NUMBER_TRAIN_IMAGES = 6000
NUMBER_TEST_IMAGES = 1000


def get_histogram():
    "create two histograms with oposite values of bins"

    first_hist = [HIGH_DIST, LOW_DIST, HIGH_DIST, LOW_DIST]
    second_hist = [LOW_DIST, HIGH_DIST, LOW_DIST, HIGH_DIST]

    assert (sum(first_hist) <= 1 or sum(second_hist) <= 1)

    return first_hist, second_hist


def create_image(hist):

    num_pixel = SIZE_IMAGE*SIZE_IMAGE
    temp_hist = num_pixel*np.array(hist)
    temp_hist = np.floor(temp_hist)

    list_values = []
    for i in range(NUM_BINS):
        ones_vec = np.ones(int(temp_hist[i]))
        single_value_vec = np.array(i*ones_vec, dtype=float)
        list_values = np.append(list_values, single_value_vec)

    return list_values


def create_dataset_premutation(hist, number_images):
    # function which creates list of images with given histogram

    list_values = create_image(hist)
    list_images = []
    for i in range(number_images):
        vec_image = np.random.permutation(list_values)
        # normalization of the vector
        vec_image = normalization(vec_image)
        list_images = np.append(list_images, vec_image)

    list_images = np.reshape(list_images, [number_images, -1])
    return list_images


def label_images():
    # create labels
    train_label0 = np.zeros([NUMBER_TRAIN_IMAGES])
    test_label0 = np.zeros([NUMBER_TEST_IMAGES])

    train_label1 = np.ones([NUMBER_TRAIN_IMAGES])
    test_label1 = np.ones([NUMBER_TEST_IMAGES])

    # create images
    hist0, hist1 = get_histogram()
    train_images0 = create_dataset_premutation(hist0, NUMBER_TRAIN_IMAGES)
    test_images0 = create_dataset_premutation(hist0, NUMBER_TEST_IMAGES)

    train_images1 = create_dataset_premutation(hist1, NUMBER_TRAIN_IMAGES)
    test_images1 = create_dataset_premutation(hist1, NUMBER_TEST_IMAGES)

    # save the plot of 3 images per each class
    # plot_image(train_images0[:3], train_label0[0])
    # plot_image(train_images1[:3], train_label1[0])
    print("Examples saved")

    train_images = np.append(train_images0, train_images1, axis=0)
    train_labels = np.append(train_label0, train_label1, axis=0)
    # train_images, train_labels = randomize(train_images, train_labels)

    test_images = np.append(test_images0, test_images1, axis=0)
    test_labels = np.append(test_label0, test_label1, axis=0)
    # test_images, test_labels = randomize(test_images, test_labels)

    return train_images, train_labels, test_images, test_labels


def normalization(x):
    # Standardization

    x_np = np.asarray(x)
    # assert (x_np.mean() == 0)
    # assert (x_np.std()==0)
    # z_scores_np = (x_np - x_np.mean()) / x_np.std()

    # Min-Max scaling

    assert (x_np.min() == 0)
    np_minmax = (x_np - x_np.min()) / (x_np.max() - x_np.min())

    return np_minmax # z_scores_np


def randomize(dataset, labels):
    # Generate the permutation index array.
    permutation = np.random.permutation(dataset.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def plot_image(image_set, label):

    fig = plt.figure()
    for i in range(len(image_set)):
        image = image_set[i, :]
        image = normalization(image)
        image = np.reshape(image, [SIZE_IMAGE, SIZE_IMAGE])
        image = (image * 255).astype('uint8')
        # plot = fig.add_subplot(1, len(image_set), i+1)
        imgplot = plt.imshow(image, cmap='gray')
        # plot.set_title("Learned co_filter filter")


# generate and save the data
tr_im, tr_lb, te_im, te_lb = label_images()

