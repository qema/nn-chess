from common import *
import sys
import numpy as np
from scipy.stats import ttest_1samp
import argparse

parser = argparse.ArgumentParser(description="Compare models")
parser.add_argument("model", nargs=2)
parser.add_argument("--interactive", action="store_true")
args = parser.parse_args()

model = PolicyModel().to(get_device())
if args.model[0] != "random":
    model.load_state_dict(torch.load("models/{}.pt".format(args.model[0]),
        map_location=get_device()))

opp_model = PolicyModel().to(get_device())
if args.model[1] != "random":
    opp_model.load_state_dict(torch.load("models/{}.pt".format(args.model[1]),
        map_location=get_device()))

with torch.no_grad():
    rewards = []
    board = chess.Board()
    for epoch in range(100):
        my_side = epoch % 2 == 0
        if args.interactive:
            print(board)
            print()
        while not board.is_game_over():
            if board.turn == my_side:
                move = choose_move(board, model, 0)
            else:
                move = choose_move(board, opp_model, 0)
            board.push(move)
            if args.interactive:
                print(board)
                print()
                input()

        reward = reward_for_side(board, my_side)
        rewards.append(reward)
        print("Game {}. Reward: {}".format(epoch, reward))

        board.reset()

print("Average reward for {}: {:.6f}".format(args.model[0], np.mean(rewards)))
print("Different from zero with p={:.4f}".format(
    ttest_1samp(rewards, 0)[1]))
