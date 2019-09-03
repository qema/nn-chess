from common import *
import argparse
import random
import multiprocessing as mp

parser = argparse.ArgumentParser(description="Train chess nn with SL")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--small", type=bool, default=False)
parser.add_argument("--n_workers", type=int, default=6)

def play_game(moves):
    data_pts = []
    board = chess.Board()
    for move in moves:
        try:
            move = move.split(".")[1]
            fen = board.fen()
            board.push_san(move)
            uci = board.peek().uci()
            data_pts.append((fen, uci))
        except ValueError:
            break
    return data_pts

def train(model, criterion, opt, boards, moves):
    model.zero_grad()
    pred = model(boards)
    loss = criterion(pred, moves)
    loss.backward()
    opt.step()
    return loss.item()

if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading data")
    game_num = 0
    data_pts = []
    all_moves = []
    with open("data/games.txt") as f:
        # burn header
        for burn in range(5):
            _ = next(f)
        for row in f:
            moves = row[:-2].split("###")[1].strip()
            if not moves: continue
            moves = moves.split(" ")
            all_moves.append(moves)
            if args.small and len(all_moves) == 100:
                break

    with mp.Pool(args.n_workers) as p:
        data_pts = p.map(play_game, all_moves)
        print("{} games".format(len(data_pts)))
        data_pts = [pt for l in data_pts for pt in l]
        print("{} moves".format(len(data_pts)))
    print()

    model = PolicyModel()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.NLLLoss()
    batch_num = 0
    while True:
        print("Batch {}".format(batch_num))
        boards, moves = zip(*random.sample(data_pts, args.batch_size))
        boards_t = states_to_tensor(boards)
        moves_t = moves_to_tensor(moves)

        loss = train(model, criterion, opt, boards_t, moves_t)
        print("Loss: {:.6f}".format(loss))
        print()

        torch.save(model.state_dict(), "models/supervised.pt")
        batch_num += 1
