#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np

cimport numpy as np


def fast_step(
    pre_board,
    pre_legal_actions,
    int agent_id,
    int action_r,
    int action_c,
    int board_size
):

    board = np.copy(pre_board)
    cdef long [:,:,:] board_view = board
    legal_actions = np.copy(pre_legal_actions)
    cdef long [:,:,:] pre_legal_actions_view = pre_legal_actions
    cdef long [:,:,:] legal_actions_view = legal_actions

    cdef int reward[2]
    cdef int done

    cdef int i, j, k
    cdef int directions[8][2]
    cdef int flag, flipped_something
    cdef int now_r, now_c
    cdef int has_action0, has_action1

    reward[0] = 0
    reward[1] = 0
    done = False

    if not _check_in_range(action_r, action_c, board_size):
        raise ValueError(f"out of board: {(action_r, action_c)}")
    if not 0 <= agent_id <= 1:
        raise ValueError(f"invalid agent_id: {agent_id}")

    if action_r == 3 and action_c == 3:
        if legal_actions_view[agent_id, 3, 3]:
            return (board, legal_actions, reward[0], reward[1], done)
        else:
            raise ValueError("cannot skip if there is possible action")

    if board_view[1-agent_id, action_r, action_c]:
        raise ValueError("cannot put a stone on opponent's stone")
    if board_view[agent_id, action_r, action_c]:
        raise ValueError("cannot put a stone on another stone")

    directions[0][:] = [1, 1]
    directions[1][:] = [1, 0]
    directions[2][:] = [1, -1]
    directions[3][:] = [0, -1]
    directions[4][:] = [-1, -1]
    directions[5][:] = [-1, 0]
    directions[6][:] = [-1, 1]
    directions[7][:] = [0, 1]

    board_view[agent_id, action_r, action_c] = 1

    flipped_something = 0
    for i in range(8):
        flag = 0
        now_r = action_r
        now_c = action_c
        for j in range(board_size):
            now_r += directions[i][0]
            now_c += directions[i][1]
            if not _check_in_range(now_r, now_c, board_size):
                break
            if board_view[1-agent_id, now_r, now_c]:
                flag = 1
            elif board_view[agent_id, now_r, now_c]:
                if flag:
                    flipped_something = 1
                    now_r = action_r
                    now_c = action_c
                    for k in range(j):
                        now_r += directions[i][0]
                        now_c += directions[i][1]
                        board_view[agent_id, now_r, now_c] = 1
                        board_view[1-agent_id, now_r, now_c] = 0
                    break
                else:
                    break
            else:
                break
    if not flipped_something:
        raise ValueError("There is no stone to flip")

    for i in range(board_size):
        for j in range(board_size):
            if board_view[0, i, j] or board_view[1, i, j]:
                legal_actions_view[0, i, j] = 0
                legal_actions_view[1, i, j] = 0
            else:
                legal_actions_view[0, i, j] = is_flippable(board_view, 0, i, j, board_size, directions)
                legal_actions_view[1, i, j] = is_flippable(board_view, 1, i, j, board_size, directions)

    has_action0 = 0
    has_action1 = 0
    for i in range(board_size):
        for j in range(board_size):
            if legal_actions_view[0, i, j]:
                has_action0 = 1
            if legal_actions_view[1, i, j]:
                has_action1 = 1

    if has_action0 == 0:
        legal_actions_view[0, 3, 3] = 1
    if has_action1 == 0:
        legal_actions_view[1, 3, 3] = 1

    if has_action0 == 0 and has_action1 == 0:
        done = True
        reward[0] = _check_wins(board_view, board_size)
        reward[1] = -reward[0]

    return (board, legal_actions, reward[0], reward[1], done)

cdef int is_flippable(long [:,:,:] board_view, int agent_id, int r, int c, int board_size, int [8][2] directions):
    cdef int i, j
    cdef int flag
    cdef int now_r, now_c

    for i in range(8):
        flag = 0
        now_r = r
        now_c = c
        for j in range(board_size):
            now_r += directions[i][0]
            now_c += directions[i][1]
            if not _check_in_range(now_r, now_c, board_size):
                break
            if board_view[1-agent_id, now_r, now_c]:
                flag = 1
            elif board_view[agent_id, now_r, now_c]:
                if flag:
                    return 1
                else:
                    break
            else:
                break
    return 0

cdef int _check_in_range(int pos_r, int pos_c, int bottom_right = 8):
    return (0 <= pos_r < bottom_right and 0 <= pos_c < bottom_right)

cdef int _check_wins(long [:,:,:] board_view, int board_size):
    cdef int i, j
    cdef int agent0_cnt = 0, agent1_cnt = 0
    for i in range(board_size):
        for j in range(board_size):
            if board_view[0, i, j]:
                agent0_cnt += 1
            elif board_view[1, i, j]:
                agent1_cnt += 1
    if agent0_cnt > agent1_cnt:
        return 1
    elif agent0_cnt < agent1_cnt:
        return -1
    return 0
