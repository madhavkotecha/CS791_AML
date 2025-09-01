import json
import heapq
import itertools
import math

########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################

class Inference:
    def __init__(self, data):
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

        self.variables_count = self.factors_count * self.num_observations
        self.messages_from_factors_to_variables = [[[1]*self.states_count for _ in range(self.factors_count)] for _ in range(self.num_observations)]
        self.messages_from_transitions_to_variables = [[[[1]*self.states_count]*2 for factor_num in range(self.factors_count)] for n in range(self.num_observations-1)]
        self.messages_from_variables_to_factors = [[[1]*self.states_count for _ in range(self.factors_count)] for _ in range(self.num_observations)]
        self.messages_from_variables_to_transitions = [[[[1]*self.states_count]*2 for factor_num in range(self.factors_count)] for n in range(self.num_observations-1)]
        
    def get_state_factor_potential(self, states, obs):
        index = 0
        for i, state in enumerate(states):
            index += state * (self.states_count ** (self.factors_count - i))
        index += obs
        return self.state_factor_potentials[index]

    def get_transition_potential(self, factor, prev_state, curr_state):
        index = prev_state * self.states_count + curr_state
        return self.transition_potentials[factor][index]

    def normalize_message(self, message):
        total = math.fsum(message)
        if total == 0:
            return [1.0/len(message)] * len(message)
        return [msg/total for msg in message]
        
    def compute_marginals(self):
        
        max_iterations = 25
        # LBP iterations
        for i in range(max_iterations):
            self.loopy_belief_propagate_to_factor()
            self.loopy_belief_propagate_to_variable()
            
        # compute marginal prob
        beliefs = []
        for t in range(self.num_observations):
            for f in range(self.factors_count):
                belief = [1.0] * self.states_count
                
                # Multiply by message from state factor
                obs_msg = self.messages_from_factors_to_variables[t][f]
                for i in range(self.states_count):
                    belief[i] *= obs_msg[i]
                
                # Multiply by messages from transition factors
                #left
                if t > 0:
                    trans_msg = self.messages_from_transitions_to_variables[t-1][f][1]
                    for i in range(self.states_count):
                        belief[i] *= trans_msg[i]

                #right
                if t < self.num_observations - 1:
                    trans_msg = self.messages_from_transitions_to_variables[t][f][0]
                    for i in range(self.states_count):
                        belief[i] *= trans_msg[i]

                beliefs.append(self.normalize_message(belief))
                
        return beliefs

    def loopy_belief_propagate_to_variable(self):
        
        # from state factors to variables
        for t in range(self.num_observations):
            obs = self.observation_sequence[t]
            
            for target_factor in range(self.factors_count):
                message = [0.0] * self.states_count

                for target_state in range(self.states_count):
                    prob_sum = 0.0
                    
                    for other_states in itertools.product(range(self.states_count),repeat=self.factors_count-1):

                        full_states = [0] * self.factors_count
                        full_states[target_factor] = target_state
                        
                        other_idx = 0
                        for f in range(self.factors_count):
                            if f != target_factor: 
                                full_states[f] = other_states[other_idx]
                                other_idx += 1

                        obs_potential = self.get_state_factor_potential(full_states, obs)

                        prob = obs_potential
                        for f in range(self.factors_count):
                            if f != target_factor:
                                prob *= self.messages_from_variables_to_factors[t][f][full_states[f]]
                        
                        prob_sum += prob
                    
                    message[target_state] = prob_sum
                
                self.messages_from_factors_to_variables[t][target_factor] = self.normalize_message(message)
        
        # from transition factors to variables
        for t in range(self.num_observations - 1):
            
            for f in range(self.factors_count):

                msg_to_curr = [0.0] * self.states_count
                for curr_state in range(self.states_count):
                    prob_sum = 0.0
                    for next_state in range(self.states_count):

                        trans_prob = self.get_transition_potential(f, curr_state, next_state)

                        next_msg = self.messages_from_variables_to_transitions[t][f][1][next_state]
                        prob_sum += trans_prob * next_msg
                    
                    msg_to_curr[curr_state] = prob_sum
                
                msg_to_next = [0.0] * self.states_count
                for next_state in range(self.states_count):
                    prob_sum = 0.0
                    for curr_state in range(self.states_count):

                        trans_prob = self.get_transition_potential(f, curr_state, next_state)

                        curr_msg = self.messages_from_variables_to_transitions[t][f][0][curr_state]
                        prob_sum += trans_prob * curr_msg
                    
                    msg_to_next[next_state] = prob_sum
                
                self.messages_from_transitions_to_variables[t][f][0] = self.normalize_message(msg_to_curr)
                self.messages_from_transitions_to_variables[t][f][1] = self.normalize_message(msg_to_next)

    def loopy_belief_propagate_to_factor(self):
        
        # from variables to state factors
        for t in range(self.num_observations):
            for f in range(self.factors_count):
                message = [1.0] * self.states_count
                
                # left (previous time step, if exists)
                if t > 0:
                    trans_msg = self.messages_from_transitions_to_variables[t-1][f][1]
                    for i in range(self.states_count):
                        message[i] *= trans_msg[i]
                
                # right (next time step, if exists)
                if t < self.num_observations - 1:
                    trans_msg = self.messages_from_transitions_to_variables[t][f][0]
                    for i in range(self.states_count):
                        message[i] *= trans_msg[i]
                
                self.messages_from_variables_to_factors[t][f] = self.normalize_message(message)
        
        # from variables to transition factors
        for t in range(self.num_observations - 1):
            for f in range(self.factors_count):

                msg_from_curr = [1.0] * self.states_count
                
                obs_msg = self.messages_from_factors_to_variables[t][f]
                for i in range(self.states_count):
                    msg_from_curr[i] *= obs_msg[i]
                
                # left (previous time step, if exists)
                if t > 0:
                    prev_trans_msg = self.messages_from_transitions_to_variables[t-1][f][1]
                    for i in range(self.states_count):
                        msg_from_curr[i] *= prev_trans_msg[i]
                
                msg_from_next = [1.0] * self.states_count
 
                obs_msg = self.messages_from_factors_to_variables[t+1][f]
                for i in range(self.states_count):
                    msg_from_next[i] *= obs_msg[i]
                
                # right (next time step, if exists)
                if t + 1 < self.num_observations - 1:
                    next_trans_msg = self.messages_from_transitions_to_variables[t+1][f][0]
                    for i in range(self.states_count):
                        msg_from_next[i] *= next_trans_msg[i]
                
                self.messages_from_variables_to_transitions[t][f][0] = self.normalize_message(msg_from_curr)
                self.messages_from_variables_to_transitions[t][f][1] = self.normalize_message(msg_from_next)
        
    def factor_in_fhmm(self):
        pass

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
            marginals = inference.compute_marginals()
            output.append({
                'Marginals': marginals,
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('TestCases.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')