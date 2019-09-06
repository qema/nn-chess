from common import *

class MinimaxAgent:
    def __init__(self, value_model, max_depth=3):
        self.value_model = value_model
        self.max_depth = max_depth

    def choose_move(self, board):
        return self.minimax(board, self.max_depth, -float("inf"), float("inf"))

    def minimax(self, board, depth, alpha, beta):
        if board.is_game_over():
            reward = 0 if board.result() == "1/2-1/2" else -100
            return None, reward
        elif depth == 0:
            #piece_values = {chess.PAWN: 1,
            #    chess.BISHOP: 3,
            #    chess.KNIGHT: 3,
            #    chess.ROOK: 5,
            #    chess.QUEEN: 9,
            #    chess.KING: 0}
            #pieces = board.piece_map().values()
            #my_pieces = [piece_values[p.piece_type]
            #    for p in pieces if p.color == board.turn]
            #their_pieces = [piece_values[p.piece_type]
            #    for p in pieces if p.color != board.turn]
            #value = sum(my_pieces) - sum(their_pieces)
            #return None, value
            board_t = states_to_tensor([board.fen()])
            return None, self.value_model(board_t).item()
        else:
            best_move, best_value = None, -float("inf")
            legal_moves = list(board.legal_moves)
            for move in legal_moves:
                board.push(move)
                _, next_value = self.minimax(board, depth-1, -beta, -alpha)
                next_value *= -1
                if next_value > best_value:
                    best_value = next_value
                    alpha = next_value
                    best_move = move
                board.pop()
                if alpha >= beta:
                    break
        return best_move, alpha

if __name__ == "__main__":
    model = ValueModel()
    model.load_state_dict(torch.load("models/value.pt",
        map_location=get_device()))
    agent = MinimaxAgent(model)

    board = chess.Board()
    while not board.is_game_over():
        if board.turn:
            move, value = agent.choose_move(board)
        else:
            move, value = agent.choose_move(board)
            #done = False
            #while not done:
            #    try:
            #        move = chess.Move.from_uci(input())
            #        done = True
            #    except ValueError:
            #        pass
        print(value)
        board.push(move)
        print(board)
