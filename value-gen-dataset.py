from common import *
import argparse
import random

parser = argparse.ArgumentParser(description="Generate value net dataset")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dataset_size", type=int, default=1000000)
parser.add_argument("--n_workers", type=int, default=6)
args = parser.parse_args()

def run_games(process_idx, batch_size, s_model, rl_model, file_lock):
    n_completed = 0
    while True:#len(all_fens) < args.dataset_size:
        print("Process {}: {} completed".format(process_idx, n_completed))
        boards = [chess.Board() for i in range(batch_size)]
        record_pts = [random.randint(0, 200) for i in range(batch_size)]
        done_idxs = set()
        fens = [None for i in range(batch_size)]
        sides = [None for i in range(batch_size)]
        rewards = [None for i in range(batch_size)]

        t = 0
        n_done = 0
        while n_done < batch_size:
            s_board_idxs, r_board_idxs = [0]*batch_size, [0]*batch_size
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
                    else:
                        actions = torch.distributions.Categorical(
                            logits=pred[valid_idxs])
                        action_idx = actions.sample()
                        move = legal_moves[action_idx.item()]
                        if move.promotion is not None:
                            move.promotion = 5
                    board.push(move)

                    if t == record_pts[n]:
                        fens[n] = board.fen()
                        sides[n] = board.turn

                    if board.is_game_over():
                        done_idxs.add(n)
                        n_done += 1
                        rewards[n] = reward_for_side(board, sides[n])
            t += 1

        with file_lock:
            with open("proc/value-net-boards.txt", "a") as f_boards:
                with open("proc/value-net-rewards.txt", "a") as f_rewards:
                    for fen, reward in zip(fens, rewards):
                        if fen is not None:
                            f_boards.write("{}\n".format(fen))
                            f_rewards.write("{}\n".format(reward))
                            n_completed += 1

if __name__ == "__main__":
    print("Note: if starting from scratch, make sure to delete existing "
        "data in proc/")

    s_model = PolicyModel().to(get_device())
    s_model.load_state_dict(torch.load("models/supervised.pt",
        map_location=get_device()))

    rl_model = PolicyModel().to(get_device())
    # TODO: reinforce
    rl_model.load_state_dict(torch.load("models/supervised.pt",
        map_location=get_device()))

    file_lock = mp.Lock()
    with torch.no_grad():
        mp.spawn(run_games, args=(args.batch_size, s_model, rl_model,
            file_lock), nprocs=args.n_workers)
