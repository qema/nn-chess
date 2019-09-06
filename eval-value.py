from common import *

policy = PolicyModel().to(get_device())
policy.load_state_dict(torch.load("models/supervised.pt",
    map_location=get_device()))
val_model = ValueModel().to(get_device())
val_model.load_state_dict(torch.load("models/value.pt",
    map_location=get_device()))

while True:
    board = chess.Board()
    while not board.is_game_over():
        board_t = states_to_tensor([board.fen()])
        print(board.turn, val_model(board_t))
        if board.turn:
            move = choose_move(board, policy, 0)
        else:
            done = False
            while not done:
                try:
                    move = chess.Move.from_uci(input())
                    done = True
                except ValueError:
                    print("bad")
                    pass

        board.push(move)
        print(board)
        #input()
