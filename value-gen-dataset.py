from common import *
import argparse
import random

parser = argparse.ArgumentParser(description="Generate value net dataset")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dataset_size", type=int, default=1024)
args = parser.parse_args()

s_model = PolicyModel().to(get_device())
s_model.load_state_dict(torch.load("models/supervised.pt",
    map_location=get_device()))

rl_model = PolicyModel().to(get_device())
# TODO: reinforce
rl_model.load_state_dict(torch.load("models/supervised.pt",
    map_location=get_device()))

all_fens, all_rewards = [], []
while len(all_fens) < args.dataset_size:
    print(len(all_fens))
    n_games = args.batch_size
    boards = [chess.Board() for i in range(n_games)]
    record_pts = [random.randint(0, 200) for i in range(n_games)]
    done_idxs = set()
    fens = [None for i in range(n_games)]
    sides = [None for i in range(n_games)]
    rewards = [None for i in range(n_games)]

    t = 0
    n_done = 0
    while n_done < n_games:
        s_board_idxs, r_board_idxs = [0]*n_games, [0]*n_games
        s_boards, r_boards = [], []
        for n, board in enumerate(boards):
            if n not in done_idxs:
                if t < record_pts[n]:
                    s_board_idxs[n] = len(s_boards)
                    s_boards.append(board.fen())
                elif t > record_pts[n]:
                    r_board_idxs[n] = len(r_boards)
                    r_boards.append(board.fen())

        boards_t = states_to_tensor(s_boards + r_boards)
        s_boards_t, r_boards_t = (boards_t[:len(s_boards)],
            boards_t[len(s_boards):])

        if s_boards:
            pred_s = s_model(s_boards_t)
        if r_boards:
            pred_r = rl_model(r_boards_t)
    
        for n, board in enumerate(boards):
            if n not in done_idxs:
                legal_moves = list(board.legal_moves)
                valid_idxs = [move_to_action_idx(move) for move in
                    legal_moves]
                if t < record_pts[n]:
                    pred = pred_s[s_board_idxs[n]]
                elif t > record_pts[n]:
                    pred = pred_r[r_board_idxs[n]]

                if t == record_pts[n]:
                    move = random.choice(legal_moves)
                    fens[n] = board.fen()
                    sides[n] = board.turn
                else:
                    actions = torch.distributions.Categorical(
                        logits=pred[valid_idxs])
                    action_idx = actions.sample()
                    move = legal_moves[action_idx.item()]
                    if move.promotion is not None:
                        move.promotion = 5
                board.push(move)

                if board.is_game_over():
                    done_idxs.add(n)
                    n_done += 1
                    rewards[n] = reward_for_side(board, sides[n])
        t += 1

    for fen, reward in zip(fens, rewards):
        if fen is not None:
            all_fens.append(fen)
            all_rewards.append(reward)

with open("proc/value-net-boards.txt", "w") as f:
    for fen in all_fens:
        f.write("{}\n".format(fen))
with open("proc/value-net-rewards.txt", "w") as f:
    for reward in all_rewards:
        f.write("{}\n".format(reward))
