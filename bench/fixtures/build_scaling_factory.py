# FLE lab-play scaling factory — POST this file's contents as the "code" field of
# /execute on an iron_ore_throughput sandbox. It is agent-namespace code (nearest(),
# place_entity(), Prototype, ... are injected by FLE), NOT a host script.
#
# Verified on fle-sandbox:bench: 9.0 s to build, entity_count 270, and
# /probe {"entity":"iron-ore"} returns 188.895 ore per 60 in-game seconds, identical
# across 4 consecutive probes (it does not saturate).
#
# Production core: burner mining drills on the nearest iron-ore patch, each dropping
# straight into its own wooden chest (800-item buffer => throughput stays flat across a
# whole probe series instead of saturating the way a belt buffer does).
# Ballast: 1x1 medium electric poles placed ON the ore patch (resource tiles carry no
# trees, so placement never fails) purely to reach a target entity count.
#
# All-or-nothing: the counts below are the fixture's contract, so a short build raises
# instead of returning. A run that quietly placed 7 of 10 drills is a DIFFERENT factory
# than the one the numbers above were measured on, and every downstream entity_count /
# throughput assertion would read that smaller build as if it were this one.
N_DRILLS = 10  # capped by the wooden chests in LAB_PLAY_POPULATED_STARTING_INVENTORY
N_BALLAST = 250  # tune this to hit the entity_count you want
COAL_EACH = 45  # ~1200 in-game seconds of fuel = ~20 probe windows

p = nearest(Resource.IronOre)
move_to(p)
bb = get_resource_patch(Resource.IronOre, p, radius=30).bounding_box
X0, Y0 = int(bb.left_top.x), int(bb.left_top.y)
X1, Y1 = int(bb.right_bottom.x), int(bb.right_bottom.y)

# A 2x2 drill at integer origin (X, Y) covers tiles (X-1..X, Y-1..Y); facing UP it drops
# onto tile (X-1, Y-2).  Everything stays inside the patch.
drill_y = Y1
drop_y = drill_y - 2

placed = 0
drill_note = ""
for x in range(X0 + 1, X1 + 1, 2):
    if placed >= N_DRILLS:
        break
    # Reach matters: place_entity and insert_item both need the character close.
    move_to(Position(x=x + 0.5, y=drill_y + 2.5))  # stand just below the drill
    try:
        d = place_entity(
            Prototype.BurnerMiningDrill,
            position=Position(x=float(x), y=float(drill_y)),
            direction=Direction.UP,
        )
    except Exception as e:
        drill_note = f"drill at x={x} failed: {type(e).__name__}: {e}"
        print(drill_note)
        continue
    try:
        insert_item(Prototype.Coal, d, COAL_EACH)
    except Exception as e:
        # An unfuelled drill mines nothing: it would inflate entity_count while
        # contributing zero ore, so it goes back in the inventory like a drill
        # with no chest does.
        drill_note = f"drill at x={x} took no coal: {type(e).__name__}: {e}"
        print(drill_note)
        pickup_entity(d)
        continue
    # Compute the drop tile explicitly: entity.drop_position can be stale after
    # insert_item and will hand you the PREVIOUS drill's tile.
    chest_pos = Position(x=x - 0.5, y=drill_y - 1.5)
    move_to(Position(x=x - 0.5, y=drop_y - 2.5))  # stand just above the chest
    try:
        place_entity(Prototype.WoodenChest, position=chest_pos)
    except Exception as e:
        drill_note = f"chest at x={x} failed: {type(e).__name__}: {e}"
        print(drill_note)
        pickup_entity(d)  # no sink -> the drill would stall, do not keep it
        continue
    placed += 1

poles = 0
pole_note = ""
for i in range(N_BALLAST):
    bx, by = X0 + (i % 24), Y0 + (i // 24)
    if by >= drop_y - 1:
        pole_note = f"patch ran out at pole {i}: row y={by} reaches the drop lane"
        break
    if i % 6 == 0:
        move_to(Position(x=bx + 0.5, y=by + 0.5))
    try:
        place_entity(Prototype.MediumElectricPole, position=Position(x=bx + 0.5, y=by + 0.5))
        poles += 1
    except Exception as e:
        pole_note = f"pole {i} at ({bx},{by}) failed: {type(e).__name__}: {e}"

print("built:", placed, "drill+chest pairs and", poles, "ballast poles")
if placed != N_DRILLS or poles != N_BALLAST:
    raise RuntimeError(
        f"scaling fixture incomplete: {placed}/{N_DRILLS} drill+chest pairs, "
        f"{poles}/{N_BALLAST} ballast poles on patch "
        f"({X0},{Y0})-({X1},{Y1}); last drill issue: {drill_note or 'none'}; "
        f"last ballast issue: {pole_note or 'none'}"
    )
