import torch
import torch.nn as nn
import torch.optim as optim
import chess
import random
import multiprocessing as mp

#import chess.variant
#chess.Board = chess.variant.RacingKingsBoard

class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.conv1 = nn.Conv2d(20, 128, 5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        #self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        #self.relu4 = nn.ReLU()
        self.fc1 = nn.Linear(64*128, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 64*64)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, boards):
        out = self.conv1(boards)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        #out = self.conv4(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        out = self.softmax(out)
        #out = out.view(out.shape[0], 2, 8, 8)
        return out

device_cache = None
def get_device():
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
    return device_cache

# input: list of fens OLD
def states_to_tensor_old(states):
    boards_t = []
    board = chess.Board()
    for state in states:
        board.set_fen(state)
        side = board.turn
        piece_map = board.piece_map()

        pieces_t = torch.zeros(12, 8, 8, device=get_device())
        for pos, piece in piece_map.items():
            col, row = chess.square_file(pos), chess.square_rank(pos)
            idx = int(piece.color != side)*6 + (piece.piece_type-1)
            pieces_t[idx][row][col] = 1

        legal_t = torch.zeros(12, 8, 8, device=get_device())
        capture_t = torch.zeros(12, 8, 8, device=get_device())
        check_t = torch.zeros(12, 8, 8, device=get_device())
        for tmp_turn in [side, not side]:
            board.turn = tmp_turn
            for move in board.legal_moves:
                piece = piece_map[move.from_square]
                to_pos = move.to_square
                col, row = chess.square_file(to_pos), chess.square_rank(to_pos)
                idx = int(tmp_turn)*6 + (piece.piece_type-1)
                legal_t[idx][row][col] = 1
                if to_pos in piece_map:
                    capture_t[idx][row][col] = 1
                board.push(move)
                if board.is_check():
                    check_t[idx][row][col] = 1
                board.pop()
        board.turn = side

        board_t = torch.cat((pieces_t, legal_t, capture_t, check_t), dim=0)
        boards_t.append(board_t)
    boards_t = torch.stack(boards_t)
    return boards_t

def state_to_tensor(state):
    board = chess.Board()
    board.set_fen(state)
    side = board.turn
    piece_map = board.piece_map()

    pieces_t = torch.zeros(12, 8, 8, device=get_device())
    for pos, piece in piece_map.items():
        col, row = chess.square_file(pos), chess.square_rank(pos)
        idx = int(piece.color != side)*6 + (piece.piece_type-1)
        pieces_t[idx][row][col] = 1

    legal_t = torch.zeros(2, 8, 8, device=get_device())
    capture_t = torch.zeros(2, 8, 8, device=get_device())
    check_t = torch.zeros(2, 8, 8, device=get_device())
    checkmate_t = torch.zeros(2, 8, 8, device=get_device())
    for tmp_turn in [side, not side]:
        board.turn = tmp_turn
        for move in board.legal_moves:
            piece = piece_map[move.from_square]
            to_pos = move.to_square
            col, row = chess.square_file(to_pos), chess.square_rank(to_pos)
            idx = int(tmp_turn)
            legal_t[idx][row][col] = 1
            if board.is_capture(move):
                capture_t[idx][row][col] = 1
            board.push(move)
            if board.is_checkmate():
                checkmate_t[idx][row][col] = 1
                check_t[idx][row][col] = 1
            elif board.is_check():
                check_t[idx][row][col] = 1
            board.pop()
    board.turn = side

    board_t = torch.cat((pieces_t, legal_t, capture_t, check_t,
        checkmate_t), dim=0)
    return board_t

# input: list of fens
def states_to_tensor(states, n_workers=1):
    boards_t = []
        #boards_t.append(board_t)
    with mp.Pool(n_workers) as pool:
        boards_t = pool.map(state_to_tensor, states)
    boards_t = torch.stack(boards_t)
    return boards_t

def action_idx_to_move(idx):
    return chess.Move(idx // 64, idx % 64)

def move_to_action_idx(move):
    return move.from_square*64 + move.to_square

# input: list of uci moves
def moves_to_tensor(moves):
    action_tensors = []
    for t, action in enumerate(moves):
        move = chess.Move.from_uci(action)
        action_tensors.append(move_to_action_idx(move))
    action_tensors = torch.tensor(action_tensors, dtype=torch.long,
        device=get_device())
    return action_tensors

def choose_move(board, model, eps):
    legal_moves = list(board.legal_moves)
    if random.random() < eps:
        move = random.choice(legal_moves)
    else:
        board_t = states_to_tensor([board.fen()])
        pred = model(board_t)

        valid_idxs = [move_to_action_idx(move) for move in legal_moves]
        pred = pred[0][valid_idxs]
        actions = torch.distributions.Categorical(logits=pred)
        move = legal_moves[actions.sample().item()]
        if move.promotion is not None:
            move.promotion = 5
    return move

reward_dict = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
# precond: game is over
def reward_for_side(board, side):
    result = board.result()
    reward = reward_dict[result]
    if not side:
        reward *= -1
    return reward

if __name__ == "__main__":
    policy = PolicyModel()
    board = chess.Board()
    boards_t = states_to_tensor([board.fen()])
    policy(boards_t)
