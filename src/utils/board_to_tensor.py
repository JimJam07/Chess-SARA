import chess.pgn
import chess
import numpy as np
import time
from tqdm import tqdm
import torch.nn as nn
import torch

def format_time(seconds):
    """Convert seconds into hours, minutes, and seconds format."""
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


class MoveEmbedding(nn.Module):
    """
    A class used to embed the one hot coded policy
    """
    def __init__(self, move_vocab_size=4672, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(move_vocab_size, embed_dim)

    def forward(self, move_index):
        return self.embed(move_index).detach()  # Outputs a 128-dim vector

def getValue(score):
    """
    Convert Stockfish evaluation to a normalized float in range [-1, 1].
    Args:
        score(stockfish): the position evaluation of stockfish

    Returns:
        value(float): the normalised Centipawn Score
    """
    if score.is_mate():
        return 1.0 if score.mate() > 0 else -1.0  # Win/Loss mapped to ±1
    else:
        return np.tanh(score.score() / 800.0)  # Normalize centipawn scores

def getPolicy(move, board, embedding):
    """
    Converts a move into a one-hot vector over all legal moves.
    Args:
        move (chess)- current chess move
        board (chess)- state of the board
        embedding (nn.Module) - a embedding neural network

    Returns:
        one_hot - a one hot coded vector of moves
    """
    move_index = None
    legal_moves = list(board.legal_moves)  # Get all legal moves
    uci_moves = [move.uci() for move in legal_moves]
    if str(move) in uci_moves:
        move_index = legal_moves.index(move)
        move_index = torch.tensor([move_index])
        embed = embedding(move_index)
        return embed.squeeze(0)

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
    state_tensor[12, :, :] = float(board.turn == chess.WHITE)

    # 3. Castling Rights (Planes 13-16)
    castling_rights = board.castling_rights
    state_tensor[13, :, :] = float(bool(castling_rights & chess.BB_H1))  # White kingside
    state_tensor[14, :, :] = float(bool(castling_rights & chess.BB_A1))  # White queenside
    state_tensor[15, :, :] = float(bool(castling_rights & chess.BB_H8)) # Black kingside
    state_tensor[16, :, :] = float(bool(castling_rights & chess.BB_A8))  # Black queenside

    # 4. En Passant Target Square (Plane 17)
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        state_tensor[17, row, col] = 1

    # **5. Half-Move Clock (Plane 18)**
    state_tensor[18, :, :] = board.halfmove_clock / 100  # Normalize for training

    return torch.from_numpy(state_tensor)

def TensorEncoder(pgn_file, limit_games = None, stockfish_path="/opt/homebrew/bin/lc0", depth=8):
    """
    Convert a PGN file into a list of Stockfish-style tensors.
    :param: pgn_file: The PGN file path
    :param: limit_games: Number of games to consider for the tensor
    :param: stockfish_path: The location of stockfish engine
    :param: depth: The depth of stockfish analysis
    :return: Tensor Encoded dataset
    """
    games = []
    embedding = MoveEmbedding()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    start_time = time.time()  # Start timing
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    print(f"Read {len(games)}....")
    if limit_games is not None:
        print(f"Limiting the games to {limit_games}....")
        np.random.shuffle(games)
        games = games[:limit_games]
    tensor_data = []
    value_data = []
    policy_data = []
    for game in tqdm(games, desc="Processing Games", unit="game"):
        board = game.board()
        for move in game.mainline_moves():
            policy = getPolicy(move, board, embedding)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))  # Analyze position
            score = info["score"].relative
            value = torch.tensor(getValue(score), dtype=torch.float32)
            value_data.append(value)
            policy_data.append(policy)
            tensor_data.append(board_to_stockfish_tensor(board))  # Convert board to tensor
            board.push(move)
    end_time = time.time()
    total_time = end_time - start_time
    formatted_time = format_time(total_time)

    print(f"Time taken: {formatted_time}")

    return torch.stack(tensor_data), torch.stack(value_data), torch.stack(policy_data)

