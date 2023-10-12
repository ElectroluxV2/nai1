from easyAI import TwoPlayerGame, Negamax, Human_Player, AI_Player

"""
    "Dodge The Pile" is a two-player turn-based game where players take alternating turns manipulating a shared score.
    The objective of the game is to strategically add or subtract numbers from the score in a way that avoids negative values.
    The available numbers for moves are shared between the players and consist of the following integers: [1, 2, 3, 4, 6, 12].

    Player 1 and Player 2, represented by AI and Human players, respectively, compete to achieve different win conditions.
    Player 1 (AI) aims to secure victory by ensuring that the final score is less than zero
    while Player 2 (Human) seeks to win by achieving a final score greater than zero.

    The game provides a variety of possible moves at each turn, and each number may be used only once throughout the game.
    The players take turns making their moves and strategically depleting the pool of available numbers.
    The game ends when one of the players achieves their respective win condition or when there are no more moves available.
 """


class DodgeThePileGame(TwoPlayerGame):

    def __init__(self, players=None):
        """Game initialization"""
        self.players = players
        self.score = 0  # Start with 0
        self.current_player = 2  # Player 2 starts (Human)
        self.available_moves = ["-1", "1", "-2", "2", "-3", "3", "-4", "4", "-6", "6", "-12", "12"]

    def possible_moves(self):
        """Returns available move options"""
        return self.available_moves

    def make_move(self, move):
        """Makes a move by updating the game state"""
        # Players have shared possible moves, each number may be added or subtracted only once
        chosen_move = str(move)
        dumped_move = str(-1 * int(chosen_move))

        if chosen_move in self.available_moves:
            self.available_moves.remove(chosen_move)

        if dumped_move in self.available_moves:
            self.available_moves.remove(dumped_move)

        self.score += int(move)  # Make move

    def win(self):
        """Checks for a winning condition"""
        # Player 1 wins when there are no possible moves and score is negative, Player 2 when score is positive
        return self.score < 0 and len(self.available_moves) == 0

    def is_over(self):
        """Checks if the game is over"""
        # game stops when someone wins or there are no more moves available
        return self.win() or len(self.available_moves) == 0

    def scoring(self):
        """Determines the game's score"""
        return 100 if self.win() else 0

    def show(self):
        """Displays the current game state"""
        print("Score: %d, possible moves: %s" % (self.score, self.available_moves))


ai1 = Negamax(5)  # The AI will think 5 moves in advance
game = DodgeThePileGame([AI_Player(ai1), Human_Player()])
history = game.play()
