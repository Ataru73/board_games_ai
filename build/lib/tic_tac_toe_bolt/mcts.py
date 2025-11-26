import numpy as np
import torch
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        terminated = False
        winner = 0
        
        # 1. Selection
        while not node.is_leaf():
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            obs, reward, terminated, truncated, info = state.step(action)
            if terminated:
                # If game ended, reward is 1 for current player (who just moved), so -1 for next player?
                # In our env, reward is 1 if the move wins.
                # So the player who made the move won.
                # The value for the NEXT player (whose turn it would be) is -1.
                # But `state.step` switches the player internally.
                # So if `terminated` is True, the game is over.
                # The value returned should be from the perspective of the player who would play next.
                # If P1 moved and won, it's now P2's turn (conceptually), but game is over.
                # P2 sees a loss (-1).
                leaf_value = -1.0 
                node.update_recursive(-leaf_value)
                return

        # 2. Expansion and Evaluation
        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples and a score for the current player.
        # Check if game is already over (e.g. draw or win just happened in the last step of selection?)
        # Actually, if we broke out of while loop because node is leaf, the game might not be over.
        
        # We need to check valid moves.
        # In our env, all 9 moves are "valid" in action space, but some are illegal (occupied).
        # The NN should learn to assign 0 prob to illegal moves.
        # But MCTS should also filter them?
        # Our env returns -10 reward for illegal move and doesn't change state.
        # We should probably mask illegal moves here.
        
        action_probs, leaf_value = self._policy(state)
        
        # Check for end of game (if state is terminal but not caught in selection loop - e.g. root was leaf)
        # But we can't easily check terminal without stepping.
        # Assuming _policy handles evaluation.
        
        node.expand(action_probs)
        
        # 3. Backup
        node.update_recursive(leaf_value) 

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        # Loop for n_playout times
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

# We need to refine _playout to work with the Gym env.
# And we need a policy_value_fn wrapper that converts the NN output to what MCTS expects.
