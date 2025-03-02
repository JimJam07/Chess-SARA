import chess
import numpy as np
import chess.pgn

def board_to_stockfish_tensor(board):
    """
    Convert a chess board into an 8×8×19 Stockfish-style tensor.
    :param: board: a chess board state at a given time
    :return: tensor: a tensor encoding of the board state
    """
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    state_tensor = np.zeros((19, 8, 8), dtype=np.float32)

    # **1. Piece Positions (Planes 0-11)**
    for square, piece in board.piece_map().items():
        piece_type = piece_map[piece.piece_type]
        channel = piece_type + (6 if piece.color == chess.BLACK else 0)
        row, col = divmod(square, 8)
        state_tensor[channel, row, col] = 1  # Mark presence of the piece

    # 2. Side to Move (Plane 12)
    state_tensor[12, :, :] = 1 if board.turn == chess.WHITE else 0

    # 3. Castling Rights (Planes 13-16)
    state_tensor[13, :, :] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    state_tensor[14, :, :] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    state_tensor[15, :, :] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    state_tensor[16, :, :] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

    # 4. En Passant Target Square (Plane 17)
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        state_tensor[17, row, col] = 1

    # **5. Half-Move Clock (Plane 18)**
    state_tensor[18, :, :] = board.halfmove_clock / 100  # Normalize for training

    return state_tensor

def TensorEncoder(pgn_file, limit_games = None):
    """
    Convert a PGN file into a list of Stockfish-style tensors.
    :param: pgn_file: The PGN file path
    :param: limit_games: Number of games to consider for the tensor
    :return: Tensor Encoded dataset
    """
    games = []
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    if limit_games is not None:
        print(f"Limiting the games to {limit_games}")
        np.random.shuffle(games)
        games = games[:limit_games]
    tensor_data = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            tensor_data.append(board_to_stockfish_tensor(board))  # Convert board to tensor

    return np.array(tensor_data)

