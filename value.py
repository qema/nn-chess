from common import *
import argparse

parser = argparse.ArgumentParser(description="Train chess nn with RL")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_workers", type=int, default=0)
args = parser.parse_args()

def train(model, criterion, opt, board, reward):
    model.zero_grad()
    pred = model(board)
    loss = criterion(pred, reward)
    loss.backward()
    opt.step()
    return loss.item()

all_boards, all_rewards = [], []
with open("proc/value-net-boards.txt", "r") as f:
    for line in f:
        all_boards.append(line[:-1])
with open("proc/value-net-rewards.txt", "r") as f:
    for line in f:
        all_rewards.append(int(line[:-1]))

model = ValueModel().to(get_device())
criterion = nn.MSELoss()
opt = optim.Adam(model.parameters())

for epoch in range(10000000):
    print("Batch {}".format(epoch))
    sample_idxs = random.sample(range(len(all_boards)), args.batch_size)
    boards, rewards = [], []
    for idx in sample_idxs:
        boards.append(all_boards[idx])
        rewards.append(all_rewards[idx])

    boards_t = states_to_tensor(boards, n_workers=args.n_workers)
    rewards_t = torch.tensor(rewards, dtype=torch.float, device=get_device())
    loss = train(model, criterion, opt, boards_t, rewards_t)

    print("Loss: {:.6f}".format(loss))
    print()

    torch.save(model.state_dict(), "models/value.pt")
