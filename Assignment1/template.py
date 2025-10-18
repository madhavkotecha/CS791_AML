import json
import math
from itertools import product
from collections import defaultdict
import heapq



########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################




class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        
        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.
        
        Refer to the sample test case for the structure of the input data.
        """
        self.factors_count = data['Factors_Count']
        self.states_count = data['State_Count']
        self.num_observations = data['Number of Observations']
        self.observation_sequence = data['Observation Sequence']
        self.transition_potentials = data['Transition Potentials']
        self.transition_potentials = list(self.transition_potentials.values())
        self.transition_potentials_sum = [[sum(self.transition_potentials[i][j:j+self.states_count]) for j in range(0, len(self.transition_potentials[i]), self.states_count)] for i in range(len(self.transition_potentials))]
        self.transition_potentials = [[potential/self.transition_potentials_sum[i][j//self.states_count] for j, potential in enumerate(self.transition_potentials[i])] for i in range(len(self.transition_potentials))]
        self.state_factor_potentials = data['State_Factor_Potentials']
        self.state_factor_potentials_sum = [sum(self.state_factor_potentials[i:i+self.states_count]) for i in range(0, len(self.state_factor_potentials), self.states_count)]
        self.state_factor_potentials = [potential/self.state_factor_potentials_sum[i//self.states_count] for i, potential in enumerate(self.state_factor_potentials)]
        self.k = data['K']
        
        self.alpha = self.forward_pass()
        self.beta = self.backward_pass()
        self.z_value = None
        # print(self.alpha)

    def triangulate_and_get_cliques(self):
        pass

    def get_junction_tree(self):
        pass

    def assign_potentials_to_cliques(self):
        pass

    def get_z_value(self):
        if self.z_value == None:
            self.z_value = sum(self.alpha[self.num_observations - 1].values())
        return self.z_value

    def compute_marginals(self):
        marginals = []
        for t in range(self.num_observations):
            for f in range(self.factors_count):
                marginal = [0.0] * self.states_count
                
                for state in range(self.states_count):
                    prob_sum = 0.0
                    
                    for config in product(range(self.states_count), repeat=self.factors_count):
                        if config[f] == state and config in self.alpha[t] and config in self.beta[t]:
                            prob_sum += self.alpha[t][config] * self.beta[t][config]
                    
                    marginal[state] = prob_sum / self.get_z_value()
                
                marginals.append(marginal)

        return marginals

    def compute_top_k(self):
        all_assignments = []
        total_vars = self.factors_count * self.num_observations
        
        for assignment in product(range(self.states_count), repeat=total_vars):
            unnormalized_prob = self.compute_assignment_prob(assignment)
            normalized_prob = unnormalized_prob / self.get_z_value()  # Normalize by Z
            all_assignments.append({
                "assignment": list(assignment),
                "probability": normalized_prob
            })
        
        # sort by prob in desc order
        all_assignments.sort(key=lambda x: x['probability'], reverse=True)
        return all_assignments[:self.k]

    def compute_assignment_prob(self, assignment):
        probability = 1.0
        
        assignment_matrix = []
        idx = 0
        for t in range(self.num_observations):
            time_states = []
            for f in range(self.factors_count):
                time_states.append(assignment[idx])
                idx += 1
            assignment_matrix.append(tuple(time_states))
        
        # initial emission probability
        obs_idx = self.get_obs_idx(assignment_matrix[0], self.observation_sequence[0])
        probability *= self.state_factor_potentials[obs_idx]
        
        # transition and emission probabilities for remaining t
        for t in range(1, self.num_observations):
            
            # transition
            for f in range(self.factors_count):
                prev_state = assignment_matrix[t-1][f]
                curr_state = assignment_matrix[t][f]
                trans_idx = prev_state * self.states_count + curr_state
                probability *= self.transition_potentials[f][trans_idx]
            
            # emission
            obs_idx = self.get_obs_idx(assignment_matrix[t], self.observation_sequence[t])
            probability *= self.state_factor_potentials[obs_idx]
        
        return probability

    def get_obs_idx(self, state_config, obs_value):
        index = 0
        for f in range(self.factors_count):
            index += state_config[f] * (self.states_count ** (self.factors_count - f))
        
        index += obs_value
        return index

    def forward_pass(self):
        K, M, T = self.states_count, self.factors_count, self.num_observations
        joint_states = list(product(range(K), repeat=M))
        alpha = [dict() for _ in range(T)]

        # t = 0
        for v in joint_states:
            idx = self.get_obs_idx(v, self.observation_sequence[0])
            alpha[0][v] = self.state_factor_potentials[idx]

        # t >= 1
        for t in range(1, T):
            partial_alpha = alpha[t-1].copy()

            # fold
            for f in range(M):
                updated_alpha = defaultdict(float)
                for prev_v, val in partial_alpha.items():
                    for new_s in range(K):
                        p = self.transition_potentials[f][prev_v[f]*K + new_s]
                        key = prev_v[:f] + (new_s,) + prev_v[f+1:]
                        updated_alpha[key] += val * p
                partial_alpha = updated_alpha

            # mult by emission
            for v in joint_states:
                idx = self.get_obs_idx(v, self.observation_sequence[t])
                alpha[t][v] = partial_alpha[v] * self.state_factor_potentials[idx]

        return alpha

    def backward_pass(self):
        K, M, T = self.states_count, self.factors_count, self.num_observations
        joint_states = list(product(range(K), repeat=M))
        beta = [dict() for _ in range(T)]

        # t = T-1
        for v in joint_states:
            beta[T-1][v] = 1.0

        # t <= T-2
        for t in range(T-2, -1, -1):
            temp_beta = {}
            for v in joint_states:
                idx = self.get_obs_idx(v, self.observation_sequence[t+1])
                temp_beta[v] = beta[t+1][v] * self.state_factor_potentials[idx]

            # fold
            for f in range(M):
                updated_beta = defaultdict(float)
                for v, val in temp_beta.items():
                    for u in range(K):
                        p = self.transition_potentials[f][u*K + v[f]]
                        key = v[:f] + (u,) + v[f+1:]
                        updated_beta[key] += val * p
                temp_beta = updated_beta
            beta[t] = temp_beta

        return beta



########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('TestCases.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')
