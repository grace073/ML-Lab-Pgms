MAX, MIN = 1000, -1000

def alphabeta_minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta, maxDepth):
    if depth == maxDepth:
        return values[nodeIndex]

    if maximizingPlayer:
        best = MIN
        for i in range(0, 2):
            val = alphabeta_minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta, maxDepth)
            best = max(best, val)
            alpha = max(alpha, best)

            if beta <= alpha:
                break
        return best

    else:
        best = MAX
        for i in range(0, 2):
            val = alphabeta_minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta, maxDepth)
            best = min(best, val)
            beta = min(beta, best)

            if beta <= alpha:
                break

        return best

values = [3, 5, 6, 9, 1, 2, 0, -1]
maxDepth = len(values).bit_length() - 1
print("The optimal value is:", alphabeta_minimax(0, 0, True, values, MIN, MAX, maxDepth))
