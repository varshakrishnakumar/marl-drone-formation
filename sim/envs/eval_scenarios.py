# sim/envs/eval_scenarios.py

EVAL_SCENARIOS = {
    # 1) Hover formation: frozen leader, no obstacles, spawn in formation
    "hover_formation": dict(
        leader_speed_scale = 0.0,
        spawn_in_formation = True,
        disable_static     = True,
        disable_dynamic    = True,
    ),

    # 2) Leader tracking: moving leader, no obstacles
    "leader_tracking": dict(
        leader_speed_scale = 0.3,   # default speed
        spawn_in_formation = True,
        disable_static     = True,
        disable_dynamic    = True,
    ),

    # 3) Obstacle field: moving leader + static + dynamic obstacles
    "obstacle_field": dict(
        leader_speed_scale = 0.3,
        spawn_in_formation = True,
        disable_static     = False,
        disable_dynamic    = False,
    ),
}
