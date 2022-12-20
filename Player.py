import numpy as np
MINIMAX_DEPTH=2
EXPECTIMAX_DEPTH=2

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def outcome(self, board, successor, player_number): # Copies the board into another array

        outcome = board.copy()
        move = successor[1]

        if 0 in outcome[:,move]:
            update_row = -1

            for row in range(1, outcome.shape[0]):
                update_row = -1

                if outcome[row, move] > 0 and outcome[row - 1, move] == 0:
                    update_row = row - 1
                elif row == outcome.shape[0] - 1 and outcome[row, move] == 0:
                    update_row = row

                if update_row >= 0:
                    outcome[update_row, move] = player_number
                    break

        return outcome 

    def successors(self, board): # Gets an array of all successor states

        successor_list = []

        for col in range(len(board[0] - 1)):
            row = 5

            while board[row][col] != 0 and row > 0:
                row = row - 1

            if board[row][col] == 0:
                successor_list.append((row,col))

        return successor_list

    def check_connected(self, board, number): # Checks for possible patterns of connected chips that can lead to a win

        next_move = -1
        successors = self.successors(board)

        for successor in successors:
            outcome = self.outcome(board, successor, number)

            if self.chance_score(outcome, number, True) > 0:
                next_move = successor[1]
                return True, next_move

        return False, next_move 

    def max_value(self, board, next_move, alpha, beta, depth): # max value implementation for minimax algorithm

        connected = self.check_connected(board, self.player_number)

        if connected[0]:  
            return True, connected[1], -999999999
        elif depth >= MINIMAX_DEPTH:
            score = self.evaluation_function(board)
            return False, next_move, score

        score = -999999999

        for successor in self.successors(board):
            outcome = self.outcome(board, successor, self.player_number)
            result = self.min_value(outcome, successor[1], alpha, beta, depth+1)

            if result[0]:
                return True, result[1], -999999999
            elif not result[0]:
                if result[2] >= score:
                    next_move = successor[1]
                score = max(score, result[2])
                if score >= beta:
                    return False, next_move, score

                alpha = max(alpha, score)

        return False, next_move, score

    def min_value(self, board, next_move, alpha, beta, depth): # min value implementation for minimax algorithm

        connected = self.check_connected(board, ((self.player_number*2)%3) )

        if connected[0]:  
            return True, connected[1], 999999999
        elif depth >= MINIMAX_DEPTH:
            score = self.evaluation_function(board)

            return False, next_move, score

        score = 999999999

        for successor in self.successors(board):
            outcome = self.outcome(board, successor, self.player_number)
            result = self.max_value(outcome, successor[1], alpha, beta, depth+1)

            if result[0]:
                return True, result[1], -999999999
            elif not result[0]:
                if result[2] >= score:
                    next_move = successor[1]
                score = min(score, result[2])
                if score <= alpha:
                    return False, next_move, score
                beta = min(beta, score)

        return False, next_move, score

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        alpha = 0
        beta = 0
        depth = 0
        next_move = 0
        result = self.max_value(board, next_move, alpha, beta, depth)

        return result[1]

        raise NotImplementedError('Whoops I don\'t know what to do')

    def expectimax_max_value(self, board, player, next_move, depth): # max value function for expectimax algorithm

        score = -999999999
        successors = self.successors(board)
        for successor in successors:
            outcome = self.outcome(board, successor, self.player_number)
            result = self.expectimax_helper(outcome, player, next_move, depth)

            if result[0] :
                return True, result[1], 999999999
            elif result[2] > score:
                next_move = successor[1]
            score = max(score, result[2])

        return False, next_move, score 

    def expected_value(self, board, player, next_move, depth): # expected value function for expectimax algorithm

        score = 999999999
        successors = self.successors(board) #pass the other player's number
        p = 1/len(successors)

        for successor in successors:
            outcome = self.outcome(board, successor, self.player_number)
            result = self.expectimax_helper(outcome, ((player*2)%3), next_move, depth)
            if result[0] :
                return True, result[1], -999999999
            else:
                score = score + p*result[2]

        return False, next_move, score

    def expectimax_helper(self, board, player, next_move, depth): # helper function for expectimax algorithm

        connected = self.check_connected(board, ((self.player_number*2)%3))

        if connected[0]:  
            return True, connected[1], 999999999
        elif depth >= EXPECTIMAX_DEPTH:
            score = self.evaluation_function(board)
            return False, next_move, score

        if player == self.player_number:
            return self.expectimax_max_value(board, player, next_move, depth+1)
        else:
            return self.expected_value(board, player, next_move, depth+1)

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        depth = 0
        next_move = 0

        result = self.expectimax_helper(board, self.player_number, next_move, depth)

        return result[1] 

        raise NotImplementedError('Whoops I don\'t know what to do')

    def chance_score(self, board, player_number, connected): # Helper function for evaluation function

        to_str = lambda a: ''.join(a.astype(str))
        BASE = 2

        def match_pattern(num_tile):

            pnum_str = '{0}'.format(player_number)
            str = ""

            for i in range(0, num_tile):
                str += pnum_str

            for i in range(0, 4 - num_tile):
                str += '0'

            return str

        def check_horizontal(b, num_tile): # from ConnectFour.py

            str = match_pattern(num_tile)

            return horizontal_helper(b, str) + horizontal_helper(b, str[::-1])

        def horizontal_helper(b, chance_str): # Helper function to implement pattern matches

            chances = 0
            for row in b:
                if chance_str in to_str(row):
                    chances += 1

            return chances
            
        def check_vertical(b, num_tile): # from ConnectFour.py
            str = match_pattern(num_tile)

            return horizontal_helper(b.T, str[::-1])

        def check_diagonal(b, num_tile): # from ConnectFour.py
            str = match_pattern(num_tile)

            return diagonal_helper(b, str[::-1])

        def diagonal_helper(b, chance_str): # Helper function to implement pattern matches, from ConnectFour.py
            chances = 0

            for op in [None, np.fliplr]:
                op_board = op(b) if op else b
                
                root_diag = np.diagonal(op_board, offset = 0).astype(np.int8)
                if chance_str in to_str(root_diag):
                    chances+=1

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset = offset)
                        diag = to_str(diag.astype(np.int8))
                        if chance_str in diag:
                            chances += 1

            return chances

        score = 0

        if connected:
            score = (check_horizontal(board, 4) +  check_vertical(board, 4) + check_diagonal(board, 4))
        elif not connected :
            for i in range(1, 4):
                score += (check_horizontal(board, i) +  check_vertical(board, i) + check_diagonal(board, i)) * pow(BASE, (i - 1))

        return score

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """

        player1 = self.chance_score(board, self.player_number, False)
        player2 = self.chance_score(board, (self.player_number * 2) % 3, False)

        utility = player2 - player1

        return utility

class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

