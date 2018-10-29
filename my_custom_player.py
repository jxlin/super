import math
import random
from sample_players import BasePlayer, DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        depth_limit = 100
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            for depth in range(1, depth_limit + 1):
                action = self.alpha_beta_search(state, depth)
                if action is not None:
                    self.queue.put(action)
            #self.queue.put(self.alpha_beta(state, depth_limit=10))
            #self.queue.put(self.minimax(state, depth=3))
    
    def minimax(self, state, depth):

        def min_val(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_val(state.result(action), depth - 1))
            return value

        def max_val(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_val(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_val(state.result(x), depth - 1))

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
    
    def alpha_beta(self, state, depth_limit=100):
        best_move = None
        for depth in range(1, depth_limit+1):
            best_move = self.alpha_beta_search(state, depth)
        return best_move
    
    def alpha_beta_search(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in state.actions():
            v = self.min_value(state.result(a), depth - 1, alpha, beta)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        file = open("result.txt", "a")
        file.write(str(depth) + ", " + str(state.ply_count) + "\n")
        file.close()
        return best_move
    
    def min_value(self, state, depth, alpha, beta):
        if depth <= 0:
            return self.my_moves(state)
        
        if state.terminal_test():
            return state.utility(self.player_id)
        
        v = float("inf")
        for a in state.actions():
            v = min(v, self.max_value(state.result(a), depth - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def max_value(self, state, depth, alpha, beta):
        if depth <= 0:
            return self.my_moves(state)
        
        if state.terminal_test():
            return state.utility(self.player_id)
        
        v = float("-inf")
        for a in state.actions():
            v = max(v, self.min_value(state.result(a), depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def moves(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_moves = state.liberties(own_loc)
        opp_moves = state.liberties(opp_loc)
        return len(own_moves) - len(opp_moves)
    
    def my_moves(self, state):
        width = 11
        height = 9
        borders = [
            [(0, _) for _ in range(width)],
            [(_, 0) for _ in range(height)],
            [(height - 1, _) for _ in range(width)],
            [(width - 1, _) for _ in range(height)]
        ]
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_moves = state.liberties(own_loc)
        opp_moves = state.liberties(opp_loc)
        own_score = 0
        opp_score = 0
        for move in own_moves:
            if self.is_in_borders(move, borders) and state.ply_count < 30:
                own_score += 10
            elif self.is_in_borders(move, borders) and state.ply_count < 50:
                own_score -= 20
            elif self.is_in_borders(move, borders):
                own_score -= 30
            else:
                own_score += 10
        
        for move in opp_moves:
            if self.is_in_borders(move, borders) and state.ply_count < 30:
                opp_score += 10
            elif self.is_in_borders(move, borders) and state.ply_count < 50:
                opp_score -= 20
            elif self.is_in_borders(move, borders):
                opp_score -= 30
            else:
                opp_score += 10
        
        return own_score - opp_score
    
    def is_in_borders(self, move, borders):
        for border in borders:
            if (move % 13, math.floor(move / 13)) in border:
                return True
        return False