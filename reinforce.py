from common import *
import random
import queue

game_batch_size = 128
max_recent_opps = 10000
pool_update_dur = 64

def train(model, opt, criterion, boards, actions, rewards):
    model.zero_grad()
    pred = model(boards)
    loss = criterion(pred, actions)
    loss *= rewards
    loss = torch.sum(loss) / game_batch_size
    loss.backward()
    opt.step()
    return loss

def run_games(n_games, queue, model, opp_model, epoch):
    moves = [[] for i in range(n_games)]
    states = [[] for i in range(n_games)]
    rewards = [[] for i in range(n_games)]
    boards = [chess.Board() for i in range(n_games)]
    n_done = 0
    done_idxs = set()
    t = 0

    while n_done < n_games:
        board_t = states_to_tensor([board.fen() for board in boards])
        if t % 2 == 0:
            pred_l = model(board_t[:n_games//2])
            pred_r = opp_model(board_t[n_games//2:])
        else:
            pred_l = opp_model(board_t[:n_games//2])
            pred_r = model(board_t[n_games//2:])
        pred = torch.cat((pred_l, pred_r), dim=0)

        for n, board in enumerate(boards):
            if not board.is_game_over():
                legal_moves = list(board.legal_moves)
                valid_idxs = [move_to_action_idx(move) for move in legal_moves]
                pred_n = pred[n][valid_idxs]
                actions = torch.distributions.Categorical(logits=pred_n)
                move = legal_moves[actions.sample().item()]
                if move.promotion is not None:
                    move.promotion = 5
                if (n < n_games//2) == (t % 2 == 0):
                    moves[n].append(move.uci())
                    states[n].append(board.fen())
                board.push(move)
            else:
                if n not in done_idxs:
                    done_idxs.add(n)
                    n_done += 1
                    reward = reward_for_side(board, n < n_games//2)
                    #print(n, board.result(), reward)
                    rewards[n] += [reward]*len(moves[n])

                    queue.put((moves[n], states[n], rewards[n]))
        t += 1


if __name__ == "__main__":
    model = PolicyModel().to(get_device())
    opp_model = PolicyModel().to(get_device())
    #opp_model.load_state_dict(torch.load("models/supervised.pt",
    #    map_location=get_device()))
    opp_model_pool = []

    opt = optim.Adam(model.parameters(), lr=1e-3)
    #opt = optim.SGD(model.parameters(), lr=1e-5)
    criterion = nn.NLLLoss(reduction="none")

    for epoch in range(10000):
        print("Epoch {}".format(epoch))
        # play n games
        moves, states, rewards = [], [], []
        with torch.no_grad():
            q = queue.Queue()
            run_games(game_batch_size, q, model, opp_model, epoch)

            while not q.empty():
                m, s, r = q.get()
                moves += m
                states += s
                rewards += r

        # train
        boards = states_to_tensor(states)
        actions = moves_to_tensor(moves)
        rewards = torch.tensor(rewards, dtype=torch.float,
            device=get_device())
        loss = train(model, opt, criterion, boards, actions, rewards)
        print("Loss: {:.6f}".format(loss.item()))
        print()

        torch.save(model.state_dict(), "models/reinforce.pt")

        if epoch % pool_update_dur == 0:
            opp_model_pool.append(model.state_dict())
            opp_model_pool = opp_model_pool[-max_recent_opps:]

        # pick random opponent out of pool
        params = random.choice(opp_model_pool)
        opp_model.load_state_dict(params)
