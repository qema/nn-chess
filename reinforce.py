from common import *
import random

game_batch_size = 128
max_recent_opps = 10000
pool_update_dur = 64
grad_clip = 0.25

def train(model, opt, criterion, boards, actions, rewards):
    model.zero_grad()
    pred = model(boards)
    loss = criterion(pred, actions)
    loss *= rewards
    loss = torch.sum(loss) / game_batch_size
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return loss.item()

def run_games(n_games, model, opp_model, epoch):
    moves = [[] for i in range(n_games)]
    states = [[] for i in range(n_games)]
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
            pred_r = opp_model(r_boards_t)
        else:
            pred_l = opp_model(l_boards_t)
            pred_r = model(r_boards_t)

        for n, board in enumerate(boards):
            if n not in done_idxs:
                legal_moves = list(board.legal_moves)
                valid_idxs = [move_to_action_idx(move) for move in
                    legal_moves]
                if n < n_games // 2:
                    pred = pred_l[l_board_idxs[n]][valid_idxs]
                else:
                    pred = pred_r[r_board_idxs[n]][valid_idxs]

                actions = torch.distributions.Categorical(logits=pred)
                move = legal_moves[actions.sample().item()]
                if move.promotion is not None:
                    move.promotion = 5
                if (n < n_games//2) == (t % 2 == 0):
                    moves[n].append(move.uci())
                    states[n].append(board.fen())
                board.push(move)

                if board.is_game_over():
                    done_idxs.add(n)
                    n_done += 1
                    reward = reward_for_side(board, n < n_games//2)
                    #print(n, board.result(), reward)
                    rewards[n] += [reward]*len(moves[n])
        t += 1

    flatten = lambda l: [x for y in l for x in y]
    return flatten(moves), flatten(states), flatten(rewards)

if __name__ == "__main__":
    model = PolicyModel().to(get_device())
    opp_model = PolicyModel().to(get_device())
    #opp_model.load_state_dict(torch.load("models/supervised.pt",
    #    map_location=get_device()))
    opp_model_pool = []

    opt = optim.Adam(model.parameters(), lr=1e-4)
    #opt = optim.SGD(model.parameters(), lr=1e-5)
    criterion = nn.NLLLoss(reduction="none")

    for epoch in range(10000):
        print("Epoch {}".format(epoch))
        # play n games
        moves, states, rewards = [], [], []
        with torch.no_grad():
            m, s, r = run_games(game_batch_size, model, opp_model, epoch)
            moves += m
            states += s
            rewards += r

        # train
        boards_t = states_to_tensor(states)
        actions_t = moves_to_tensor(moves)
        rewards_t = torch.tensor(rewards, dtype=torch.float,
            device=get_device())
        loss = train(model, opt, criterion, boards_t, actions_t, rewards_t)
        print("Loss: {:.6f}".format(loss))
        print()

        torch.save(model.state_dict(), "models/reinforce.pt")

        if epoch % pool_update_dur == 0:
            opp_model_pool.append(model.state_dict())
            opp_model_pool = opp_model_pool[-max_recent_opps:]

        # pick random opponent out of pool
        params = random.choice(opp_model_pool)
        opp_model.load_state_dict(params)
