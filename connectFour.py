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
                print({-1: '🔴  ', 0: ' . ', 1: '🔵  '}[i], end="", flush=True),
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
                    self.move(int(input('player 1\'s select row from 0...%i: ' % self.dim[0])), 1)
                    self.printState()
                    break
                except (NameError, ValueError, IndexError):
                    print('invalid move, try again')
            while not (self.finished):
                try:
                    self.move(int(input('player 2\'s select row from 0...%i: ' % self.dim[0])), -1)
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


class agent:

    def __init__(self, layer, board):
        layer = np.concatenate(([board.dim[0] * board.dim[1]], layer, [board.dim[0]]))
        self.w = [np.random.rand(layer[i], layer[i+1])-1 for i in range(len(layer)-1)]
        for i in range(len(self.w)-1):
            self.w[i+1] = np.dot(self.w[i], self.w[i+1])

    def makeMove(self, board):
        prob = 1/(1+np.exp(-1*np.dot(board.state.reshape(board.dim[0]*board.dim[1]), self.w[-1])))
        prob[np.nonzero(board.state[:, -1])] = 0
        prob = prob/np.sum(prob)
        return(np.random.choice(board.dim[0],  p=prob))

    def mutate(self):
        return()


def compete(agents):
    ret = np.zeros(len(agents))
    for i in range(len(agents)):
        [a1, a2] = agents[i]
        if a1 != a2:
            ret[i] = board_1.playAvA(a1, a2)
    return(ret)


board_1 = board()
a = 100
agentScore = np.zeros(a)
agentArr = [agent([8], board_1) for i in range(a)]
agentMat = [[[agentArr[i], agentArr[j]] for i in range(a)] for j in range(a)]
t0 = time.clock()
p = mp.Pool(a)
out = np.array(p.map(compete, agentMat))
agentScore = out.sum(axis=0) - out.sum(axis=1)
t1 = time.clock()
print(t1 - t0)
print(agentScore, np.sum(agentScore))
board_1.playHvH()
