# CHESS-SARA



## Introduction

This repository focuses on creating chess engine **S.A.R.A - Strategic Adaptive Reevaluation Algorithm** which is a Deep RL
algorithm which uses *CNN* along with *Proximal Policy approximation*

## Data Preparation


The data used is obtained from [FICS Game Database](https://www.ficsgames.org/download.html). Processed games from 2024 for version-1

![Chess board](./src/readme/board.png)

- We Preprocessed nearly 70k games using *Stockfish-Style Representation* in which we create (19, 8, 8 ) tensor:
- 11 represents the piece position for both white(lower number) and black(upper number)
   - 0 , 6 -  Pawns
   - 1,  7 -  knights
   - 2, 8 - Bishops
   - 3, 9 - Rooks
   - 4, 10 - Queens
   - 5, 11 - King
- 12 represents whose turn it is **1 if white, 0 if black**
- 13 - 16 represents castling rights for each king:
   - 13, 15 - King and Queen side castling for the White king
   -  14, 16 - King and Queen side castling for the Black king
- 17 represents *En Passant* opportunity
- 18 ensure we enforce the *[50 move rule](https://en.wikipedia.org/wiki/Fifty-move_rule#:~:text=The%20fifty%2Dmove%20rule%20in,the%20opponent%20completing%20a%20turn)*

This Gives much more information, such as the historical context to train the model with but it has its own pros and cons

### Pros

---

✅ Encodes additional game features (castling, en passant, move history).

✅ More expressive for reinforcement learning (good for AlphaZero-style models).

✅ Side-to-move plane helps with learning asymmetry in move choices.

### Cons

---

❌ Larger tensor size (8×8×19 = 1216 values per position).

❌ More computationally expensive tha traditional 12 channel tensors (*Bitboards*).

❌ Still lacks deeper history, e.g., past N board states for move sequence modeling.
