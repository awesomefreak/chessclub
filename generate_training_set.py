#!/usr/bin/env python3
import os
import chess
import chess.pgn
import chess.engine
import numpy as np
from state import State
import math

def get_dataset(num_samples=None):
  engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
  X,Y = [], []
  gn = 0
  m = 0
  # pgn files in the data folder
  for fn in os.listdir("data"):
    pgn = open(os.path.join("data", fn))
    while len(X) <= num_samples:
      #print("OldGame", gn)
      game = chess.pgn.read_game(pgn)
      print("NewGame", gn)
      print(f"/{game.mainline_moves()}/")
      board = game.board()
      for i, move in enumerate(game.mainline_moves()):
        #print(i, move)
        board.push(move)
        ser = State(board).serialize()
        #print(boasrd)
        res = engine.analyse(board, chess.engine.Limit(depth=10))
        print(res["score"].white().score(mate_score=100000))
        print(math.atan(res["score"].white().score(mate_score=100000)/200)*2/math.pi)
        X.append(ser)
        Y.append(math.atan(res["score"].white().score(mate_score=100000))*2/math.pi)
        m = max(m, i)
      print("parsing game %d, got %d examples" % (gn, len(X)))
      gn += 1
  engine.quit()
  print(m)
  X = np.array(X)
  Y = np.array(Y)
  return X,Y

if __name__ == "__main__":
  X,Y = get_dataset(100000)
  np.savez("processed/dataset_100K.npz", X, Y)

