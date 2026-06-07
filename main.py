from collections import deque
import time

FACTORY, SCOUT, WORKER, MINER = 0, 1, 2, 3
DIRS = ("NORTH", "EAST", "WEST", "SOUTH")
OFFSETS = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}
WALL_BITS = {"NORTH": 1, "EAST": 2, "SOUTH": 4, "WEST": 8}
MAX_ENERGY = {SCOUT: 100, WORKER: 300, MINER: 500}
COMBAT_RANK = {FACTORY: 4, MINER: 3, WORKER: 2, SCOUT: 1}
STATE_BY_PLAYER = {}


def fresh_state(player):
    return {"player": player, "step": -1, "southBound": 0, "known_walls": {},
            "known_nodes": {}, "known_mines": {}, "explored": set(), "assigned_targets": {}}


def parse_pos_key(key):
    try:
        c, r = str(key).split(",", 1)
        return int(c), int(r)
    except:
        return None


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def robot_record(uid, data):
    v = list(data) + [0] * 8
    return {"uid": str(uid), "type": int(v[0]), "col": int(v[1]), "row": int(v[2]),
            "energy": int(v[3]), "owner": int(v[4]), "move_cd": int(v[5]),
            "jump_cd": int(v[6]), "build_cd": int(v[7])}


def step_pos(p, d, n=1):
    return (p[0] + OFFSETS[d][0] * n, p[1] + OFFSETS[d][1] * n)


def agent(obs, config):
    start_time = time.time()  # 改名：避免和BFS的start冲突
    p = int(obs.get("player", 0))
    w = int(getattr(config, "width", 20))
    step = int(obs.get("step", 0))
    south = int(obs.get("southBound", 0))
    north = south + int(getattr(config, "height", 20)) - 1
    act_timeout = getattr(config, "actTimeout", 1.0) * 0.9

    state = STATE_BY_PLAYER.get(p)
    if state is None or step < state.get("step", -1) or south < state.get("southBound", 0):
        state = fresh_state(p)
    STATE_BY_PLAYER[p] = state
    state["step"] = step
    state["southBound"] = south

    robots = {}
    factory = None
    enemy_robots = {}
    for uid, d in obs.get("robots", {}).items():
        r = robot_record(uid, d)
        if r["owner"] == p:
            robots[uid] = r
            if r["type"] == FACTORY:
                factory = r
        else:
            enemy_robots[(r["col"], r["row"])] = r

    occupied = {(r["col"], r["row"]): (uid, r["type"], r["owner"]) for uid, r in
                {**robots, **{u: robot_record(u, d) for u, d in obs.robots.items()}}.items()}
    reserved = set()
    actions = {}
    decided = {}

    walls_obs = obs.get("walls", [])
    for idx, val in enumerate(walls_obs):
        val = int(val)
        if val < 0: continue
        pos = (idx % w, south + idx // w)
        state["known_walls"][pos] = val
        state["explored"].add(pos)

    def wall_at(pos):
        return state["known_walls"].get(pos, 0)

    def in_bounds(pos):
        return 0 <= pos[0] < w and south <= pos[1] <= north

    def can_step(pos, d):
        if d not in OFFSETS: return False
        np = step_pos(pos, d)
        if not in_bounds(np):
            return False
        return not (wall_at(pos) & WALL_BITS[d])

    def is_safe_move(uid, rt, pos):
        if pos in reserved:
            return False
        if pos not in occupied:
            return True
        o_uid, o_rt, o_own = occupied[pos]
        if o_uid == uid:
            return True
        if o_own == p:
            if decided.get(o_uid) in OFFSETS:
                return True
            return o_rt == SCOUT
        return COMBAT_RANK[rt] > COMBAT_RANK[o_rt]

    def safe_move(uid, rt, d):
        np = step_pos((robots[uid]["col"], robots[uid]["row"]), d)
        if not can_step((robots[uid]["col"], robots[uid]["row"]), d):
            return False
        return is_safe_move(uid, rt, np)

    # ✅ 修复后的 BFS：变量名不冲突，时间计算正确
    def bfs(start_pos, goals, max_depth=12):
        if not goals:
            return None
        # 记录BFS开始时间（关键修复）
        bfs_start = time.time()
        q = deque([(start_pos, None)])
        seen = {start_pos}

        # 正确的时间判断
        while q and time.time() - bfs_start < act_timeout:
            cur, first = q.popleft()
            if cur in goals and first:
                return first
            if len(seen) > 400:
                break
            for d in DIRS:
                if not can_step(cur, d):
                    continue
                nxt = step_pos(cur, d)
                if nxt in seen:
                    continue
                if nxt in occupied and nxt != start_pos:
                    continue
                seen.add(nxt)
                q.append((nxt, first or d))
        return None

    crystals = {}
    for k, v in obs.get("crystals", {}).items():
        pos = parse_pos_key(k)
        if pos and in_bounds(pos):
            crystals[pos] = int(v)

    nodes = set()
    for k in obs.get("miningNodes", {}):
        pos = parse_pos_key(k)
        if pos:
            nodes.add(pos)
            state["known_nodes"][pos] = step

    mines = {}
    for k, v in obs.get("mines", {}).items():
        pos = parse_pos_key(k)
        if pos:
            mines[pos] = v
            state["known_mines"][pos] = v
            nodes.discard(pos)

    frontiers = set()
    for pos in state["known_walls"]:
        for d in DIRS:
            if not (wall_at(pos) & WALL_BITS[d]):
                nxt = step_pos(pos, d)
                if in_bounds(nxt) and nxt not in state["known_walls"]:
                    frontiers.add(pos)
                    break

    ordered = sorted(robots.items(),
                     key=lambda x: (0 if x[1]["type"] == SCOUT else 1 if x[1]["type"] == WORKER else 2 if x[1][
                                                                                                              "type"] == MINER else 3))

    for uid, r in ordered:
        rt, x, y, e = r["type"], r["col"], r["row"], r["energy"]
        mv_cd, jp_cd, bd_cd = r["move_cd"], r["jump_cd"], r["build_cd"]
        gap = y - south
        pos = (x, y)

        if rt == FACTORY:
            emergency = gap <= 2 and south > 0
            north_blocked = wall_at(pos) & WALL_BITS["NORTH"]
            spawn = (x, y + 1)

            if north_blocked and jp_cd <= 1:
                jp_pos = step_pos(pos, "NORTH", 2)
                if in_bounds(jp_pos) and jp_pos not in reserved and jp_pos not in occupied:
                    actions[uid] = "JUMP_NORTH"
                    decided[uid] = "JUMP_NORTH"
                    reserved.add(jp_pos)
                    continue

            if mv_cd <= 1:
                target_row = min(north, y + 8)
                goals = {(cx, tr) for cx in range(w) for tr in range(y + 2, target_row + 1) if
                         (cx, tr) in state["explored"]}
                if not goals:
                    goals = {(cx, tr) for cx in range(w) for tr in range(y + 1, target_row + 1) if
                             (cx, tr) in state["explored"]}

                move = bfs(pos, goals, 15)
                if move and move != "SOUTH" and safe_move(uid, rt, move):
                    np = step_pos(pos, move)
                    actions[uid] = move
                    decided[uid] = move
                    reserved.add(np)
                    continue

                side = "EAST" if x < w // 2 else "WEST"
                dirs = ["NORTH", side, "WEST" if side == "EAST" else "EAST"] if not emergency else ["NORTH", side,
                                                                                                    "WEST" if side == "EAST" else "EAST",
                                                                                                    "SOUTH"]
                for d in dirs:
                    if safe_move(uid, rt, d):
                        np = step_pos(pos, d)
                        actions[uid] = d
                        decided[uid] = d
                        reserved.add(np)
                        break
                else:
                    actions[uid] = "IDLE"
                    decided[uid] = "IDLE"
                    reserved.add(pos)
                continue

            cnt_worker = sum(1 for r in robots.values() if r["type"] == WORKER)
            cnt_scout = sum(1 for r in robots.values() if r["type"] == SCOUT)
            cnt_miner = sum(1 for r in robots.values() if r["type"] == MINER)
            worker_cost = getattr(config, "workerCost", 200)
            scout_cost = getattr(config, "scoutCost", 50)
            miner_cost = getattr(config, "minerCost", 300)

            build = None
            if bd_cd <= 1 and can_step(pos, "NORTH") and spawn not in occupied and spawn not in reserved:
                if cnt_worker < 1 and e >= worker_cost + 200:
                    build = "BUILD_WORKER"
                elif nodes and cnt_miner < 1 and e >= miner_cost + 500:
                    build = "BUILD_MINER"
                elif cnt_scout < 2 and e >= scout_cost + 800:
                    build = "BUILD_SCOUT"

            if build:
                actions[uid] = build
                decided[uid] = build
                reserved.add(pos)
                reserved.add(spawn)
                continue

            actions[uid] = "IDLE"
            decided[uid] = "IDLE"
            reserved.add(pos)
            continue

        if mv_cd > 1:
            actions[uid] = "IDLE"
            decided[uid] = "IDLE"
            reserved.add(pos)
            continue

        if rt == WORKER:
            if wall_at(pos) & WALL_BITS["NORTH"] and y + 1 <= north:
                cost = getattr(config, "wallRemoveCost", 100)
                if e >= cost + 50:
                    actions[uid] = "REMOVE_NORTH"
                    decided[uid] = "REMOVE_NORTH"
                    reserved.add(pos)
                    continue

        if rt == MINER:
            if pos in nodes and e >= getattr(config, "transformCost", 100) + 50:
                actions[uid] = "TRANSFORM"
                decided[uid] = "TRANSFORM"
                reserved.add(pos)
                continue

        goals = []
        cap = MAX_ENERGY.get(rt, 9999)
        if cap - e > 10:
            goals = sorted(crystals.keys(), key=lambda p: (manhattan(pos, p), -crystals[p]))
        elif rt == SCOUT:
            goals = list(frontiers)[:20]
        elif rt == MINER:
            goals = sorted(nodes, key=lambda p: manhattan(pos, p))

        move = bfs(pos, goals, 12) if goals else None
        if move and safe_move(uid, rt, move):
            np = step_pos(pos, move)
            actions[uid] = move
            decided[uid] = move
            reserved.add(np)
            continue

        side = "EAST" if x < w // 2 else "WEST"
        for d in ["NORTH", side, "WEST" if side == "EAST" else "EAST", "SOUTH"]:
            if safe_move(uid, rt, d):
                np = step_pos(pos, d)
                actions[uid] = d
                decided[uid] = d
                reserved.add(np)
                break
        else:
            actions[uid] = "IDLE"
            decided[uid] = "IDLE"
            reserved.add(pos)

    return actions