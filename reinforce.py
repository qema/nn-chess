from common import *
import random
import argparse

parser = argparse.ArgumentParser(description="Train chess nn with RL")
parser.add_argument("--game_batch_size", type=int, default=64)
parser.add_argument("--max_recent_opps", type=int, default=10000)
parser.add_argument("--pool_update_dur", type=int, default=64)
parser.add_argument("--grad_clip", type=float, default=1)
parser.add_argument("--should_clip_grad", type=bool, default=False)
args = parser.parse_args()

def train(model, opt, log_probs, rewards):
    model.zero_grad()
    loss = -log_probs * rewards
    loss = torch.sum(loss) / args.game_batch_size
    loss.backward()
    if args.should_clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
    return loss

def run_games(n_games, model, opp_model, epoch):
    log_probs = [None for i in range(n_games)]
    rewards = [[] for i in range(n_games)]
    boards = [chess.Board() for i in range(n_games)]
    n_done = 0
    done_idxs = set()
    t = 0

    while n_done < n_games:
        l_board_idxs, r_board_idxs = [0]*n_games, [0]*n_games
        l_boards, r_boards = [], []
        for n, board in enumerate(boards):
            if n not in done_idxs:
                if n < n_games // 2:
                    l_board_idxs[n] = len(l_boards)
                    l_boards.append(board.fen())
                else:
                    r_board_idxs[n] = len(r_boards)
                    r_boards.append(board.fen())

        if l_boards:
            l_boards_t = states_to_tensor(l_boards)
        if r_boards:
            r_boards_t = states_to_tensor(r_boards)

        if t % 2 == 0:
            pred_l = model(l_boards_t)
            pred_r = opp_model(r_boards_t).detach()
        else:
            pred_l = opp_model(l_boards_t).detach()
            pred_r = model(r_boards_t)

        for n, board in enumerate(boards):
            if n not in done_idxs:
                legal_moves = list(board.legal_moves)
                valid_idxs = [move_to_action_idx(move) for move in
                    legal_moves]
                if n < n_games // 2:
                    pred = pred_l[l_board_idxs[n]]
                else:
                    pred = pred_r[r_board_idxs[n]]

                actions = torch.distributions.Categorical(
                    logits=pred[valid_idxs])
                action_idx = actions.sample()
                move = legal_moves[action_idx.item()]
                if move.promotion is not None:
                    move.promotion = 5
                if (n < n_games//2) == (t % 2 == 0):
                    log_prob = pred[valid_idxs[action_idx]]
                    if log_probs[n] is None:
                        log_probs[n] = log_prob.unsqueeze(0)
                    else:
                        log_probs[n] = torch.cat((log_probs[n],
                            log_prob.unsqueeze(0)))
                board.push(move)

                if board.is_game_over():
                    done_idxs.add(n)
                    n_done += 1
                    reward = reward_for_side(board, n < n_games//2)
                    #print(n, board.result(), reward)
                    rewards[n] += [reward]*len(log_probs[n])
        t += 1

    rewards = torch.tensor([x for l in rewards for x in l], dtype=torch.float,
        device=get_device())
    return torch.cat(log_probs), rewards

if __name__ == "__main__":
    model = PolicyModel().to(get_device())
    model.load_state_dict(torch.load("models/supervised.pt",
        map_location=get_device()))
    opp_model = PolicyModel().to(get_device())
    opp_model.load_state_dict(torch.load("models/supervised.pt",
        map_location=get_device()))
    opp_model_pool = []

    opt = optim.Adam(model.parameters(), lr=1e-3)
    #opt = optim.SGD(model.parameters(), lr=1e-5)

    for epoch in range(10000):
        print("Epoch {}".format(epoch))
        # play n games
        log_probs, rewards = run_games(args.game_batch_size, model, opp_model,
            epoch)

        # train
        loss = train(model, opt, log_probs, rewards)
        print("Loss: {:.6f}".format(loss.item()))
        print()

        torch.save(model.state_dict(), "models/reinforce.pt")

        if epoch % args.pool_update_dur == 0:
            opp_model_pool.append(model.state_dict())
            opp_model_pool = opp_model_pool[-args.max_recent_opps:]

        # pick random opponent out of pool
        params = random.choice(opp_model_pool)
        opp_model.load_state_dict(params)
