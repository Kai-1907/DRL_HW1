import numpy as np

def run_value_iteration(grid_size, start_pos, end_pos, obstacles, gamma=0.9, theta=1e-4):
    """
    執行價值迭代演算法 (Value Iteration)
    """
    # 1. 初始化價值矩陣 V
    V = np.zeros((grid_size, grid_size))
    
    # 定義獎勵：走一步扣 0.1，撞牆扣 1.0，到達終點給 20.0
    # 先設定終點的基本價值，作為價值傳遞的源頭
    V[end_pos[0]][end_pos[1]] = 20.0 
    
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上, 下, 左, 右
    action_names = ['↑', '↓', '←', '→']

    # 2. 價值迭代迴圈 (Value Iteration)
    while True:
        delta = 0
        new_V = V.copy()
        for r in range(grid_size):
            for c in range(grid_size):
                # 終點與障礙物不需要更新價值
                if (r, c) == end_pos or (r, c) in obstacles:
                    continue
                
                v_list = []
                for dr, dc in actions:
                    nr, nc = r + dr, c + dc
                    
                    # 檢查是否在邊界內且不是障礙物
                    if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in obstacles:
                        # 正常移動：獎勵 + 衰減後的下一格價值
                        reward = 20.0 if (nr, nc) == end_pos else -0.1
                        v_list.append(reward + gamma * V[nr][nc])
                    else:
                        # 撞牆或撞障礙物：留在原地並給予懲罰
                        reward = -1.0
                        v_list.append(reward + gamma * V[r][c])
                
                max_v = max(v_list)
                delta = max(delta, abs(max_v - V[r][c]))
                new_V[r][c] = max_v
        
        V = new_V
        if delta < theta:
            break

    # --- 重要：在導出策略前，確保終點和障礙物的數值是正確的 ---
    V[end_pos[0]][end_pos[1]] = 20.0 # 確保回傳給前端的是 20.00 而非 0
    for (obs_r, obs_c) in obstacles:
        V[obs_r][obs_c] = 0.0 # 障礙物顯示 0

    # 3. 導出最佳策略 (Policy Extraction)
    policy_map = {}
    for r in range(grid_size):
        for c in range(grid_size):
            # 終點標記
            if (r, c) == end_pos:
                policy_map[f"{r}-{c}"] = "END"
                continue
            # 障礙物不顯示箭頭
            if (r, c) in obstacles:
                policy_map[f"{r}-{c}"] = ""
                continue
            
            # 核心邏輯：找出鄰居中能讓 V(s') 最大的動作
            best_val = -float('inf')
            best_arrow = "↑"
            
            for i, (dr, dc) in enumerate(actions):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    current_v = V[nr][nc]
                else:
                    current_v = -float('inf') # 邊界外不考慮
                
                if current_v > best_val:
                    best_val = current_v
                    best_arrow = action_names[i]
            
            policy_map[f"{r}-{c}"] = best_arrow
            
    V[end_pos[0]][end_pos[1]] = 20.0
    return V.tolist(), policy_map
    

# HW1-2 隨機策略 
def run_policy_evaluation(grid_size, end_pos, obstacles, gamma=0.9, theta=1e-4):
    V = np.zeros((grid_size, grid_size))
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上下左右
    action_names = ['↑', '↓', '←', '→']
    
    # --- 步驟 A: 隨機生成策略 ---
    policy_map = {}
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) == end_pos:
                policy_map[f"{r}-{c}"] = "END"
            elif (r, c) in obstacles:
                policy_map[f"{r}-{c}"] = ""
            else:
                policy_map[f"{r}-{c}"] = np.random.choice(action_names)

    # --- 步驟 B: 策略評估迭代 ---
    while True:
        delta = 0
        new_V = V.copy()
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) == end_pos or (r, c) in obstacles:
                    continue
                
                act_name = policy_map[f"{r}-{c}"]
                if not act_name: continue
                
                dr, dc = actions[action_names.index(act_name)]
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in obstacles:
                    reward = 20.0 if (nr, nc) == end_pos else -0.1
                    v_next = reward + gamma * V[nr][nc]
                else:
                    v_next = -1.0 + gamma * V[r][c] # 撞牆
                
                new_V[r][c] = v_next
                delta = max(delta, abs(new_V[r][c] - V[r][c]))
        
        V = new_V
        if delta < theta:
            break

    V[end_pos[0]][end_pos[1]] = 20.0
    return V.tolist(), policy_map