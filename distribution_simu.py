import numpy as np
import matplotlib.pyplot as plt


def cum_delete(arr, b):
    cum_sum_result = (np.cumsum(arr) > b)
    if True in cum_sum_result:
        first_cannot_distribution_nbr = np.where(cum_sum_result == True)[0][0]
        arr1 = arr[:first_cannot_distribution_nbr]
        arr1 = np.append(arr1, b - sum(arr1))
        arr1 = np.append(arr1, [0] * (len(arr) - len(arr1)))
    else:
        arr1 = arr
    return arr1


def self_update(test, kits):
    kits_state1 = np.maximum(kits - (1 - test), 0)
    test = test + kits - kits_state1
    kits = kits_state1
    return test, kits


class SecondaryDistribution:

    def __index__(self, adj, key_nodes, K, max_epoch):
        """
        :param adj: adj matrix [array]
        :param key_nodes: key nodes [list]
        :param K: number of kits for key nodes
        :param max_epoch: maximum simulation length
        """

        self.adj = adj
        self.key_nodes = key_nodes
        self.K = K
        self.N = len(self.adj)  # number of nodes in the network
        self.max_epoch = max_epoch

    def distribution_kits_Poisson(self, kits):
        r = np.zeros(self.N)
        can_distribute_nodes = np.nonzero(kits)[0]
        remain_kits_can = kits[can_distribute_nodes]
        r1 = list(map(lambda mu: np.random.poisson(lam=mu), remain_kits_can))
        r[can_distribute_nodes] = r1
        r = np.minimum(np.array(r), kits)
        return r

    def secondary_distribution_process(self):
        # ----initialization--------------
        test_state = np.zeros(self.N)

        kits_state = np.zeros(self.N)
        kits_state[self.key_nodes] = self.K

        have_tested_nodes_num = [0]
        # -----------------------------------
        for t in range(1, self.max_epoch + 1):
            test_state, kits_state = self_update(test_state, kits_state)
            have_tested_nodes_num.append(sum(test_state))
            if len(np.nonzero(self.adj[np.where(test_state == 0)[0]])[1]) == 0:
                break
            un_test_nbrs_num = np.sum(self.adj, axis=1) - np.sum([list(test_state)] * self.N * self.adj, axis=1)
            un_test_nbrs_wanted_kits = np.array(list(map(lambda mu: np.random.poisson(lam=mu), un_test_nbrs_num)))
            remain_un_test_nbrs_wanted_kits = np.array(un_test_nbrs_wanted_kits)
            willing_distributed_kits = self.distribution_kits_Poisson(kits_state)
            unwilling_distributed_kits = kits_state - willing_distributed_kits
            willing_distributed_nodes = np.nonzero(willing_distributed_kits)[0]
            received_kits = np.zeros(self.N)

            for i in range(len(willing_distributed_nodes)):
                node = willing_distributed_nodes[i]
                kits = willing_distributed_kits[node]
                nbrs = np.nonzero(self.adj[node])[0]
                nbrs_test_state = test_state[nbrs]
                nbrs_received_kits = received_kits[nbrs]
                can_receive_nbrs_index = ((1 - nbrs_test_state) * (1 - np.sign(nbrs_received_kits))) > 0
                can_receive_nbrs = nbrs[can_receive_nbrs_index]
                if len(can_receive_nbrs) > kits:
                    can_receive_nbrs = can_receive_nbrs[:int(kits)]
                received_kits[can_receive_nbrs] = 1
                willing_distributed_kits[node] -= len(can_receive_nbrs)
                if willing_distributed_kits[node] > 0:
                    nbrs_wanted_kits = remain_un_test_nbrs_wanted_kits[nbrs]
                    nbrs_got_kits = cum_delete(nbrs_wanted_kits, willing_distributed_kits[node])
                    willing_distributed_kits[node] -= sum(nbrs_got_kits)
                    received_kits[nbrs] += nbrs_got_kits
                    remain_un_test_nbrs_wanted_kits[nbrs] = nbrs_wanted_kits - nbrs_got_kits

            kits_state = unwilling_distributed_kits + willing_distributed_kits + received_kits

        plt.plot(np.array(have_tested_nodes_num) / self.N)
        plt.show()

        return have_tested_nodes_num
