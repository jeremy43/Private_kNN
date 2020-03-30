# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



import numpy as np
from six.moves import xrange
import config as config


config = config.config

def labels_from_probs(probs):
    """
    Helper function: computes argmax along last dimension of array to obtain
    labels (max prob or max logit value)
    :param probs: numpy array where probabilities or logits are on last dimension
    :return: array with same shape as input besides last dimension with shape 1
            now containing the labels
    """
    # Compute last axis index
    last_axis = len(np.shape(probs)) - 1

    # Label is argmax over last dimension
    labels = np.argmax(probs, axis=last_axis)

    # Return as np.int32
    return np.asarray(labels, dtype=np.int32)



def aggregation_knn( labels, gaussian_scale, count_zero_list = None, return_clean_votes=False):
    """
      This aggregation mechanism takes the label output of knn
      shape is num_query * k, where k from knn
      resulting from inference on identical inputs and computes the most frequent
      label. It is deterministic (no noise injection like noisy_max() above.
      :param logits: logits or probabilities for each sample
      gaussian_scale: sigma for gaussian noise
      :return the most frequent label:
    """
    print('gaussian scale=',gaussian_scale)
    print('labels shape',labels.shape)
    labels_shape = np.shape(labels)
    result = np.zeros(labels_shape[0])
    nb_label = config.nb_labels
    clean_votes = np.zeros([labels_shape[0], nb_label])
    #print('count_zero_listshape in aggre', count_zero_list.shape)
    if config.dataset == 'celeba' or config.dataset=='market':
        result = np.zeros([labels_shape[0], nb_label])
        # will not return clean_votes but results
        for i in xrange(int(labels_shape[0])):
            # both count_zero or count_one shall be 40 dim, the sum of each entry = num of teachers
            
            if config.use_tau == True:
                count_zero = count_zero_list[i]
            else:
                count_zero = config.nb_teachers * np.ones(config.nb_labels) - labels[i]
            count_one = labels[i]
            #print('count_one',count_one)
            for j in range(config.nb_labels):
                count_zero[j] += np.random.normal(scale=gaussian_scale)
                count_one[j] += np.random.normal(scale=gaussian_scale)

                if count_zero[j] > count_one[j]:
                    result[i, j] = 0
                else:
                    result[i, j] = 1
        # it's not appliable for confidence based methods, which requires the margin exists for every attribute
        idx_keep = np.where(result[:, 0] >= 0)  # return idx of the number of queries
        result = np.asarray(result, dtype=np.int32)

        print('len of idx_keep', len(result))
        return idx_keep, result

    #below for cifar10
    for i in xrange(int(labels_shape[0])):
        label_count = np.bincount(labels[i, :], minlength=10)
        clean_votes[i] = np.bincount(labels[i, :], minlength=10)
        for item in xrange(10):
            label_count[item] += np.random.normal(scale=gaussian_scale)
            label_counts = np.asarray(label_count, dtype=np.float32)
        result[i] = np.argmax(label_counts)

    # print('clean_vote',clean_votes.shape)
    results = np.asarray(result, dtype=np.int32)
    clean_votes = np.array(clean_votes, dtype=np.int32)
    # confident = true mean return confident based result, only max voting greater than threshold

    if config.confident == True:
        max_list = np.max(clean_votes, axis=1)

        for i in range(len(labels)):
            max_list[i] += np.random.normal(scale=config.sigma1)
        idx_keep = np.where(max_list > config.threshold)
        idx_remain = np.where(max_list <config.threshold)
        release_vote = clean_votes[idx_keep]
        confi_result = np.zeros(len(idx_keep[0]))
        for idx, i in enumerate(release_vote):
            # print('release_vote',release_vote[idx])
            for item in range(nb_label):
                release_vote[idx, item] += np.random.normal(scale=config.gau_scale)
            # print('release_vote',release_vote[idx])
            confi_result[idx] = np.argmax(release_vote[idx])
        confi_result = np.asarray(confi_result, dtype=np.int32)
        return idx_keep, confi_result, idx_remain


    idx_keep = np.where(results >0)
    return idx_keep, results


    if return_clean_votes:
        # Returns several array, which are later saved:
        # result: labels obtained from the noisy aggregation
        # clean_votes: the number of teacher votes assigned to each sample and class
        # labels: the labels assigned by teachers (before the noisy aggregation)
        return result, clean_votes, labels
    else:
        # Only return labels resulting from noisy aggregation
        return result


def aggregation_most_frequent(logits):
    """
    This aggregation mechanism takes the softmax/logit output of several models
    resulting from inference on identical inputs and computes the most frequent
    label. It is deterministic (no noise injection like noisy_max() above.
    :param logits: logits or probabilities for each sample
    :return:
    """
    # Compute labels from logits/probs and reshape array properly
    # labels = labels_from_probs(logits)
    labels = logits
    labels_shape = np.shape(labels)
    labels = labels.reshape((labels_shape[0], labels_shape[1]))

    # Initialize array to hold final labels
    result = np.zeros(int(labels_shape[1]))

    # Parse each sample
    for i in xrange(int(labels_shape[1])):
        # Count number of votes assigned to each class
        label_counts = np.bincount(labels[:, i], minlength=10)

        label_counts = np.asarray(label_counts, dtype=np.int32)

        # Result is the most frequent label
        result[i] = np.argmax(label_counts)

    return np.asarray(result, dtype=np.int32)
