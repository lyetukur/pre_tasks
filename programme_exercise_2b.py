#define viterbi class
class Viterbi(object):
    def __init__(self, init_Prob, trans_Prob, obs_Prob,alph,states):  #initialise variables, just as in constructor
        self.init_Prob = init_Prob
        self.trans_Prob = trans_Prob
        self.obs_Prob = obs_Prob
        self.alphabet = alph
        self.states = states
        self.N = init_Prob.shape[0]

    def getStates(self, x):    # get column number of  nucleotide  in emission_matrix (i.e., here obs)
         for i, a in enumerate(self.alphabet):
              if (a == x):
                  return i

    def findHidden(self, obs):
        state_emit = np.zeros((self.N, len(obs)))   # assign a matrix with dimension: number of hidden states X length of sequence of observations

        # initialization
        state_emit[:, 0] = np.squeeze(self.init_Prob *self.obs_Prob[:,self.getStates(obs[0]),None])  # assign first state based on initial state probabilities
        for t in xrange(1, len(obs)):
             for s in xrange(0, len(self.states)):
                state_emit[s, t, None]= np.max(state_emit[:, t-1, None].T*self.trans_Prob[:,s]) * self.obs_Prob[s,self.getStates(obs[t]),None]    #recursive viterbi step
        print state_emit  # print hidden state probabilities against observed sequence
        seq=[]
        for i in xrange(0, len(obs)):
                seq.append(self.states[state_emit.argmax(0)[i]])  # capture the pointer index corresponding to optimal hidden state path
                                                                  # if there is tie in values of hidden state probabilties, fisrt occurance is reported
        return ''.join(seq)


# import numpy library for computations
import numpy as np
# initial state probabilities
pi_init = np.array([[0.5, 0.5]]).T
# transition probabilites
trans = np.array([ \
    [0.5, 0.5],\
    [0.4, 0.6]])
# observation/emission probabilities
obs = np.array([[0.2,0.3,0.3,0.2], \
                [0.3,0.2,0.2,0.3]])

alph = ('A', 'C', 'G', 'T')
states = ('H','L')
b = Viterbi(pi_init, trans, obs,alph,states)

data_ex1 = ('GGCACTGAA') # output: HHHLLLHLL (there is a tie i.e., at position 7  so output can as well be :HHHLLLLLL)
data_ex2 = ('CAAGTCCGT') # output: HLLLLHHHL
print b.findHidden(data_ex1)

