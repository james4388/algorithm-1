#!/usr/bin/env python
__author__ = 'tamnguyen'
__version__ = '1.0'

import random

class TicTocToe(object):
    """docstring for TicTocToe"""
    
    # Constant value moves
    X = 'X'
    O = 'O'
    
    # Init game with player and AI letter [X, O]
    # and board size
    def __init__(self, size=3):
        self.size = size
        self.initBoard(size)
        self.initGame()
    
    # Init game
    # Choose who play first and move
    # First player will use X
    def initGame(self):
        val = random.randint(0, 1)
        if val:
            self.player, self.ai = self.X, self.O
            self.next_move = self.player
            print "You play first: X"
        else:
            self.player, self.ai = self.O, self.X
            self.next_move = self.ai
            print "You play second O"
            
    # Initialize board game sizexsize
    # Store all moves in 2d array
    def initBoard(self, size):
        self.board = [[None]*size for i in range(size)]
        self.num_moves = 0
        
        
    # Draw board
    def drawBoard(self):
        for i in range(self.size):
            for j in range(self.size):
                letter = self.board[i][j] if self.board[i][j] else str(i*self.size + j + 1)
                print '| {}'.format(letter),
            print "|"
            print " --- "*self.size
        print "\n"
    
    # Check if game ended
    # One of player wins or
    # out of moves
    def checkGameEnded(self):
        win = self.checkWin()
        if win:
            print "{} Wins".format(win)
            return True
        elif self.num_moves == self.size**2:
            print "Draw"
            return True
        return False
            
    # Check if player or AI wins
    def checkWin(self):
        return None
    
    # Make auto move
    def doAutoMove(self):
        pos = random.randint(1, self.size**2)
        self.makeMove(pos, self.ai)
        
        
    # Make a game move by seleting position
    # from 1 to size*size
    # X or O letter
    def makeMove(self, pos, move):
        if pos < 1 or pos > self.size**2:
            return False
            
        i, j = (pos - 1)/self.size, (pos - 1)%self.size
        if not self.board[i][j]:
            self.board[i][j] = move
            self.num_moves += 1
            if self.next_move == self.player:
                self.next_move = self.ai
            else:
                self.next_move = self.player 
            return True
        else:
            return False
    
    # Play game
    def play(self):
        if self.next_move == self.player:
            m = raw_input('Select your next move .... ')
            try:
                m = int(m)
            except ValueError, e:
                print 'Invalid input'
                return False
            val = self.makeMove(m, self.player)
            if not val:
                print 'Please choose a valid position'
                return False
            return True
        else:
            self.doAutoMove()
            return True
    
    @staticmethod
    def playGame():
        game = TicTocToe(size=4)
        while not game.checkGameEnded():
            if game.play():
                game.drawBoard()
    

TicTocToe.playGame()