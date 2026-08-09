# umbra-bench

A shadow-art dataset where the same target is solved three ways.

Each sample pairs a target with the shadows produced for it:

| field | what it is |
| --- | --- |
| `prompt` | text description of the target |
| `target` | input image (the silhouette to cast) |
| `shadow_hand` | shadow made by a human, by hand |
| `shadow_teleop` | shadow made by a human teleoperating the robot fleet |
| `shadow_optimizer` | shadow produced by the optimizer |

## Why three sources

Comparing them separates two things that are usually confounded:

- **hand vs teleop** — what the robot's embodiment costs, since a human is driving in both cases
- **teleop vs optimizer** — what the algorithm costs, since the hardware is the same in both cases

A target the optimizer misses but a teleoperating human hits is an algorithm problem.
One that both miss is a reachability problem.

## Status

Early. Collecting data.
