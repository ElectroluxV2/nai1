from easyAI import TwoPlayerGame, Negamax, Human_Player, AI_Player


class DodgeThePileGame(TwoPlayerGame):
    """In turn, the players add or removes one number from score one, possible numbers are shared between players.
    Each number may be used only once. Available numbers are: [1, 2, 3, 4, 6, 12]
    Player 1 wins if final score is less than 0, Player 2 wins if final score is greater than 0"""

    def __init__(self, players=None):
        self.players = players
        self.score = 0  # Start with 0
        self.current_player = 2  # Player 2 starts (Human)
        self.available_moves = ["-1", "1", "-2", "2", "-3", "3", "-4", "4", "-6", "6", "-12", "12"]

    def possible_moves(self):
        return self.available_moves

    def make_move(self, move):
        # Players have shared possible moves, each number may be added or subtracted only once
        chosen_move = str(move)
        dumped_move = str(-1 * int(chosen_move))

        if chosen_move in self.available_moves:
            self.available_moves.remove(chosen_move)

        if dumped_move in self.available_moves:
            self.available_moves.remove(dumped_move)

        self.score += int(move)  # Make move

    def win(self):
        # Player 1 wins when there are no possible moves and score is negative, Player 2 when score is positive
        return self.score < 0 and len(self.available_moves) == 0

    def is_over(self):
        # game stops when someone wins or there are no more moves available
        return self.win() or len(self.available_moves) == 0

    def scoring(self):
        return 100 if self.win() else 0

    def show(self):
        print("Score: %d, possible moves: %s" % (self.score, self.available_moves))


ai1 = Negamax(5)  # The AI will think 5 moves in advance
game = DodgeThePileGame([AI_Player(ai1), Human_Player()])
history = game.play()
