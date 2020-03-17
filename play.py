#!/usr/bin/env python3
from __future__ import print_function
import os
import chess
import time
import chess.svg
import traceback
import base64
from state import State
import time
import torch

class Valuator(object):
  def __init__(self):
    self.count = 0
    import torch
    from train import Net
    vals = torch.load("nets/value100K.pth", map_location=lambda storage, loc: storage)
    self.model = Net()
    self.model.load_state_dict(vals)

  def __call__(self, s):
    self.count += 1
    brd = s.serialize()[None]
    output = self.model(torch.tensor(brd).float())
    #print(output.data[0][0])
    return float(output.data[0][0])


MAXVAL = 1000

def computer_minimax(s, v, depth, a, b, big=False):
  if depth >= 2 or s.board.is_game_over():
    return v(s)
  # white is maximizing player
  turn = s.board.turn
  if turn == chess.WHITE:
    ret = -MAXVAL
  else:
    ret = MAXVAL
  if big:
    bret = []

  # can prune here with beam search
  isort = []
  for e in s.board.legal_moves:
    s.board.push(e)
    isort.append((v(s), e))
    s.board.pop()
  move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)

  # beam search beyond depth 2
  if depth >= 2:
    move = move[:10]

  for e in [x[1] for x in move]:
    s.board.push(e)
    tval = computer_minimax(s, v, depth+1, a, b)
    s.board.pop()
    if big:
      bret.append((tval, e))
    if turn == chess.WHITE:
      ret = max(ret, tval)
      a = max(a, ret)
      if a >= b:
        break
    else:
      ret = min(ret, tval)
      b = min(b, ret)
      if a >= b:
        break
  if big:
    return ret, bret
  else:
    return ret

def explore_leaves(s, v):
  ret = []
  start = time.time()
  startc = v.count 
  bval = v(s)
  cval, ret = computer_minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)
  eta = time.time() - start
  print("%.2f -> %.2f: explored %d nodes in %.3f seconds %d/sec" % (bval, cval, v.count - startc , eta, int((v.count - startc)/eta)))
  return ret

# chess board and "engine"
s = State()
v = Valuator()

def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')

from flask import Flask, Response, request
app = Flask(__name__)

@app.route("/")
def hello():
  ret = open("index.html").read()
  return ret.replace('start', s.board.fen())


def computer_move(s, v):
  move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=not s.board.turn)
  if len(move) == 0:
    return
  print("top 3:")
  for i,m in enumerate(move[0:3]):
    print(i,"  ",m)
  print(s.board.turn, "moving", move[0][1])
  s.board.push(move[0][1])

@app.route("/selfplay")
def selfplay():
  s = State()

  ret = '<html><head>'
  # self play
  while not s.board.is_game_over():
    computer_move(s, v)
    ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % to_svg(s)
  print(s.board.result())

  return ret

# moves given as coordinates of piece moved
@app.route("/move_coordinates")
def move_coordinates():
  if not s.board.is_game_over():
    source = int(request.args.get('from', default='0'))
    target = int(request.args.get('to', default='0'))
    promotion = True if request.args.get('promotion', default='') == 'true' else False

    move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move)
        #print(v(s))
        if s.board.is_game_over():
          print("GAME IS OVER")
          response = app.response_class(
            response="game over",
            status=200
          )
          return response
        computer_move(s, v)
      except Exception:
        traceback.print_exc()
    response = app.response_class(
      response=s.board.fen(),
      status=200
    )
    return response

  print("GAME IS OVER")
  response = app.response_class(
    response="game over",
    status=200
  )
  return response

@app.route("/newgame")
def newgame():
  s.board.reset()
  response = app.response_class(
    response=s.board.fen(),
    status=200
  )
  return response


if __name__ == "__main__":
  if os.getenv("SELFPLAY") is not None:
    s = State()
    while not s.board.is_game_over():
      computer_move(s, v)
      print(s.board)
    print(s.board.result())
  else:
    app.run(debug=True)

