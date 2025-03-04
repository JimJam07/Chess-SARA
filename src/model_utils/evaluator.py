import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import chess.pgn
import chess
from src.utils.board_to_tensor import board_to_stockfish_tensor
from datetime import datetime
import chess


class PolicyDecoder(nn.Module):
    def __init__(self, embed_dim=128, move_vocab_size=4652):
        super().__init__()
        self.decoder = nn.Linear(embed_dim, move_vocab_size)  # Output logits for 4652 moves

    def forward(self, policy_embedding):
        """
        Decodes policy embedding into a probability distribution over 4652 moves.

        Args:
            policy_embedding (torch.Tensor): Shape (1, embed_dim), the output of the policy head.

        Returns:
            torch.Tensor: Probabilities over 4652 moves (shape: (4652,))
        """
        move_logits = self.decoder(policy_embedding)  # Shape: (1, 4652)
        move_probs = torch.softmax(move_logits, dim=-1)  # Convert to probabilities
        return move_probs.squeeze(0)  # Shape: (4652,)


def select_move_from_policy(policy_embedding, board):
    """
    Selects the best move based on the decoded policy distribution.

    Args:
        policy_embedding (torch.Tensor): Shape (1, embed_dim), the output from the policy head.
        board (chess.Board): Current board state.
        decoder (nn.Module): Decoder model to map policy embedding to 4652-move space.
        move_to_index (dict): Mapping of UCI move strings to 4652-move space indices.

    Returns:
        chess.Move: The selected move.
    """

    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]  # Convert legal moves to UCI format
    num_legal_moves = len(legal_moves)  # Number of legal moves

    # Decode policy to full 4652 move probabilitie
    decoder = PolicyDecoder(move_vocab_size=num_legal_moves)
    move_probs = decoder(policy_embedding)  # Shape: (4652,)

    # Get all legal moves in the current position

    if not legal_moves_uci:
        return None  # No legal moves available (shouldn't happen in normal play)

    # Find the best move (legal move with highest probability)
    argmax_index = torch.argmax(move_probs)

    # Convert index back to UCI move
    best_move_uci = legal_moves_uci[argmax_index]

    return chess.Move.from_uci(best_move_uci)


def convert_to_pgn(moves, event="Casual Game", site="?", date=None, round="1", white="White", black="Black",
                   result="*"):
    if date is None:
        date = datetime.today().strftime("%Y.%m.%d")

    pgn_header = f"""[Event "{event}"]
[Site "{site}"]
[Date "{date}"]
[Round "{round}"]
[White "{white}"]
[Black "{black}"]
[Result "{result}"]
"""

    pgn_moves = ""
    for i in range(0, len(moves), 2):
        move_number = i // 2 + 1
        pgn_moves += f"{move_number}. {moves[i]} "
        if i + 1 < len(moves):
            pgn_moves += f"{moves[i + 1]} "

    pgn_moves = pgn_moves.strip() + f" {result}\n"

    return pgn_header + "\n" + pgn_moves

def evaluate(model, stockfish_path="/opt/homebrew/bin/stockfish", skill=5):
    """
    A function which evaluates the strength of our model by playing with stockfish
    Args:
        model (torch.nn.Module): the trained model
        stockfish_path (str): path to the stockfish
        skill (int): the skill level of stockfish

    Returns:
        None
    """
    model.eval()
    moves = []
    # create engine and set its skill level
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": 5})
    vals = []
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:  # Model plays as White
            state_tensor = board_to_stockfish_tensor(board).unsqueeze(0).unsqueeze(0).to("cpu")  # Shape (1, 1, 19, 8, 8)
            val, policy = model(state_tensor)  # Get move embedding
            vals.append(val)
            move = select_move_from_policy(policy.squeeze(0).squeeze(0), board)  # Convert policy output to a legal move
        else:  # Stockfish plays as Black
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move
        moves.append(move)
        board.push(move)  # Make the move
    pgn = convert_to_pgn(moves)
    vals = torch.stack(vals).squeeze(1).squeeze(1)
    return pgn, vals
