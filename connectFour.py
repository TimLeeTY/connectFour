import numpy as np
import time
import multiprocessing as mp


class board:

    def __init__(self, dim=[7, 6]):
        self.dim = dim
        self.state = np.zeros(dim)
        self.finished = False

    def checkWin(self, play, row):
        colSum = self.state[:, row].cumsum()
        rowSum = self.state[play, :].cumsum()
        uDiagSum = self.state.diagonal(row - play).cumsum()
        dDiagSum = np.flip(self.state, 1).diagonal(-(row) + (self.dim[0] - play - 2)).cumsum()
        for csum in [colSum, rowSum, uDiagSum, dDiagSum]:
            if len(csum) > 3:
                csum[4:] = csum[4:] - csum[:-4]
                if np.amax(np.abs(csum[3:])) == 4:
                    return(True)
        return(False)

    def move(self, play, player):
        self.moves += 1
        t0 = time.clock()
        if self.state[play, -1] != 0 or play > self.dim[0]:
            raise NameError('invalid move')
        for row in range(self.dim[1]):
            t1 = time.clock()
            if self.state[play, row] == 0:
                self.state[play, row] = player
                if self.checkWin(play, row):
                    self.finished = True
                    self.winner = player
                break

    def printState(self):
        for row in np.flip(self.state.T, 0):
            for i in row:
                print({-1: ' x ', 0: ' . ', 1: ' o '}[i], end="", flush=True),
            print('')

    def playAvA(self, a1, a2):
        self.state = np.zeros(self.dim)
        self.finished = False
        self.moves = 0
        while self.moves < self.dim[0]*self.dim[1] and not self.finished:
            self.move(a1.makeMove(self), 1)
            self.move(a2.makeMove(self), -1)
        if not self.finished:
            self.winner = 0
        return(self.winner)

    def playHvH(self):
        self.state = np.zeros(self.dim)
        self.finished = False
        self.moves = 0
        self.printState()
        while not (self.finished) and self.moves < self.dim[0]*self.dim[1]:
            while not (self.finished):
                try:
                    self.move(int(input('player 1\'s turn, select row from 0...%i: ' % (self.dim[0] - 1))), 1)
                    self.printState()
                    break
                except (NameError, ValueError, IndexError):
                    print('invalid move, try again')
            while not (self.finished):
                try:
                    self.move(int(input('player 2\'s turn, select row from 0...%i: ' % (self.dim[0] - 1))), -1)
                    self.printState()
                    break
                except (NameError, ValueError, IndexError):
                    print('invalid move, try again')
        if not self.finished:
            self.winner = 0
            print('draw')
            self.printState()
        else:
            print('player %i won' % {1: 1, -1: 2}[self.winner])
        return(self.winner)

    def playHvA(self, a):
        self.state = np.zeros(self.dim)
        self.finished = False
        self.moves = 0
        self.printState()
        while not (self.finished) and self.moves < self.dim[0]*self.dim[1]:
            while not (self.finished):
                try:
                    self.move(int(input('player\'s turn, select row from 0...%i: ' % (self.dim[0] - 1))), 1)
                    self.printState()
                    break
                except (NameError, ValueError, IndexError):
                    print('invalid move, try again')
            self.move(a.makeMove(self), -1)
            self.printState()
        if not self.finished:
            self.winner = 0
            print('draw')
            self.printState()
        else:
            print('%s won' % {1: 'player', -1: 'computer'}[self.winner])
        return(self.winner)


class agent:

    def __init__(self, layer, board):
        layer = np.concatenate(([board.dim[0] * board.dim[1]], layer, [board.dim[0]]))
        self.w = np.array([np.random.rand(layer[i], layer[i+1])*2-1 for i in range(len(layer)-1)])

    def makeMove(self, board):
        if np.count_nonzero(board.state) == 0:
            return(np.random.choice(board.dim[0]))
        prob = np.maximum(np.dot(board.state.flatten(), self.w[0]), 0)
        for i in range(1, len(self.w)):
            prob = np.maximum(np.dot(prob, self.w[i]), 0)
        prob[np.nonzero(board.state[:, -1])] = 0
        if np.sum(prob) != 0:
            prob = prob/np.sum(prob)
            return(np.random.choice(board.dim[0], p=prob))
        else:
            return(np.random.choice(np.arange(board.dim[0])[board.state[:, -1] == 0]))

    def mutate(self, prob):
        for each in range(len(self.w)):
            temp = self.w[each].flatten()
            for i in range(len(temp)):
                if np.random.rand() < prob:
                    temp[i] = (np.random.rand())
            self.w[each] = temp.reshape(self.w[each].shape)


def compete(agents):
    ret = np.zeros(len(agents))
    for i in range(len(agents)):
        [a1, a2] = agents[i]
        if not np.array_equal(a1.w[0], a2.w[0]):
            for trials in range(10):
                ret[i] += board_1.playAvA(a1, a2)
    return(ret)


def generation(mutateProb, scoreSort, agentArr):
    agentPairs = np.array([[agentArr[scoreSort[i]], agentArr[scoreSort[j]]]
                           for i in range(len(scoreSort)) for j in range(i)])
    for pair in range(len(agentPairs)):
        [a1, a2] = agentPairs[pair]
        if np.random.rand() < mutateProb:
            a1 = agent([8], board_1)
            break
        if not (np.array_equal(a1.w[0], a2.w[0]) and np.array_equal(a1.w[1], a2.w[1])):
            for i in range(len(a1.w)):
                a1.w[i] = (a1.w[i] + a2.w[i]) / 2
        agentArr[pair] = a1
    return(agentArr)


board_1 = board()
a = 105
t0 = time.clock()
agentArr = [agent([8], board_1) for i in range(a)]
p = mp.Pool(a)
for gens in range(30):
    mutateProb = 1 / (2+gens)
    print(gens)
    agentScore = np.zeros(a)
    agentMat = [[[agentArr[i], agentArr[j]] for j in range(a)] for i in range(a)]
    out = np.array(p.map(compete, agentMat))
    agentScore = out.sum(axis=0) - out.sum(axis=1)
    scoreSort = agentScore.argsort()[-15:][::-1]
    agentArr = generation(mutateProb, scoreSort, agentArr)
t1 = time.clock()
print(t1 - t0)
print(agentScore, np.sum(agentScore))
while(True):
    board_1.playHvA(agentArr[agentScore.argmax()])
